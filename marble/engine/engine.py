# marble/engine/engine.py

"""
The core engine module that coordinates agents within the environment.
"""
import json
import os
import re
import ast
from typing import Any, Dict, List, Optional, Union

from marble.agent import BaseAgent
from marble.configs.config import Config
from marble.engine.engine_planner import EnginePlanner
from marble.environments import (
    BaseEnvironment,
    CodingEnvironment,
    DBEnvironment,
    MinecraftEnvironment,
    ResearchEnvironment,
    WebEnvironment,
    WorldSimulationEnvironment,
)
from marble.evaluator.evaluator import Evaluator
from marble.graph.agent_graph import AgentGraph
from marble.memory.base_memory import BaseMemory
from marble.memory.shared_memory import SharedMemory
from marble.utils.logger import get_logger

EnvType = Union[
    BaseEnvironment,
    WebEnvironment,
    ResearchEnvironment,
    WorldSimulationEnvironment,
    MinecraftEnvironment,
    DBEnvironment,
    CodingEnvironment,
]
AgentType = Union[BaseAgent]


class Engine:
    """
    The Engine class orchestrates the simulation, coordinating agents and the environment.
    """

    def _read_code_from_file(self, file_path: str) -> str:
        """
        Read code from a specified file path.

        Args:
            file_path (str): File path

        Returns:
            str: File content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except IOError as e:
            self.logger.error(f"Failed to read code from {file_path}: {e}")
            return ""

    def _extract_code_from_text(self, text: str) -> str:
        """
        Extract the best-effort Python code snippet from a model text output.
        """
        if not text:
            return ""

        candidates = [text]
        try:
            decoded_text = bytes(text, "utf-8").decode("unicode_escape")
            if decoded_text and decoded_text != text:
                candidates.append(decoded_text)
        except Exception:
            pass

        for candidate in candidates:
            json_blocks = re.findall(
                r"```json\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE
            )
            for json_block in json_blocks:
                extracted = self._extract_solution_from_json_text(json_block)
                if extracted:
                    return extracted

            extracted_from_candidate = self._extract_solution_from_json_text(candidate)
            if extracted_from_candidate:
                return extracted_from_candidate

            fenced_blocks = re.findall(
                r"```python\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE
            )
            if fenced_blocks:
                return max((block.strip() for block in fenced_blocks), key=len, default="")

            generic_blocks = re.findall(r"```\s*(.*?)```", candidate, re.DOTALL)
            if generic_blocks:
                for block in sorted(
                    (block.strip() for block in generic_blocks),
                    key=len,
                    reverse=True,
                ):
                    nested_code = self._extract_code_from_text(block)
                    if nested_code:
                        return nested_code
                    if block:
                        return block

            for marker in ("# file_name_", "# solution.py"):
                marker_index = candidate.find(marker)
                if marker_index != -1:
                    return candidate[marker_index:].strip()

            function_marker = "Result from the function:"
            if function_marker in candidate:
                function_part = candidate.split(function_marker, 1)[1].strip()
                json_start = function_part.find("{")
                json_end = function_part.rfind("}")
                if json_start != -1 and json_end > json_start:
                    payload = function_part[json_start : json_end + 1]
                    for parser in (json.loads, ast.literal_eval):
                        try:
                            parsed = parser(payload)
                            if isinstance(parsed, dict):
                                code_field = parsed.get("code")
                                if isinstance(code_field, str) and code_field.strip():
                                    return code_field.strip()
                        except Exception:
                            continue

            for pattern in (
                r'"code"\s*:\s*"((?:\\.|[^"\\])*)"',
                r"'code'\s*:\s*'((?:\\.|[^'\\])*)'",
            ):
                code_field_matches = re.findall(pattern, candidate, re.DOTALL)
                for matched in code_field_matches:
                    try:
                        decoded = bytes(matched, "utf-8").decode("unicode_escape")
                        if decoded.strip():
                            return decoded.strip()
                    except Exception:
                        continue

        return ""

    def _extract_solution_from_json_text(self, text: str) -> str:
        """
        Parse JSON-like text and recover solution code from known fields.
        """
        if not text:
            return ""

        raw_candidates = [text.strip()]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            raw_candidates.append(text[start : end + 1])

        for raw in raw_candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(raw)
                except Exception:
                    continue

                extracted = self._extract_solution_from_json_obj(parsed)
                if extracted:
                    return extracted

        return ""

    def _extract_solution_from_json_obj(self, obj: Any) -> str:
        """
        Recover solution code from parsed JSON object.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and key.strip().lower() == "solution.py":
                    return self._coerce_solution_value(value)

            collected_snippets = self._collect_code_snippets(obj)
            if collected_snippets:
                return "\n\n".join(collected_snippets).strip()

            for value in obj.values():
                extracted = self._extract_solution_from_json_obj(value)
                if extracted:
                    return extracted

        elif isinstance(obj, list):
            for item in obj:
                extracted = self._extract_solution_from_json_obj(item)
                if extracted:
                    return extracted

        return ""

    def _coerce_solution_value(self, value: Any) -> str:
        """
        Normalize solution field value to plain text code.
        """
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            snippets = self._collect_code_snippets(value)
            if snippets:
                return "\n\n".join(snippets).strip()
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value).strip()

    def _collect_code_snippets(self, node: Any) -> List[str]:
        """
        Recursively collect values of keys named 'code'.
        """
        snippets: List[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and key.strip().lower() == "code":
                    if isinstance(value, str) and value.strip():
                        snippets.append(value.strip())
                snippets.extend(self._collect_code_snippets(value))
        elif isinstance(node, list):
            for item in node:
                snippets.extend(self._collect_code_snippets(item))
        return snippets

    def _extract_solution_from_iterations(self, summary_data: Dict[str, Any]) -> str:
        """
        Try to recover solution code from iteration task results when no file was written.
        """
        iterations = summary_data.get("iterations", [])
        if not isinstance(iterations, list):
            return ""

        for iteration in reversed(iterations):
            if not isinstance(iteration, dict):
                continue

            iteration_summary = iteration.get("summary")
            if isinstance(iteration_summary, str):
                code_from_summary = self._extract_code_from_text(iteration_summary)
                if code_from_summary:
                    return code_from_summary

            task_results = iteration.get("task_results", [])
            if not isinstance(task_results, list):
                continue

            for item in reversed(task_results):
                text_candidates: List[str] = []
                if isinstance(item, dict):
                    if "result" in item and isinstance(item["result"], str):
                        text_candidates.append(item["result"])
                    for value in item.values():
                        if isinstance(value, str):
                            text_candidates.append(value)
                elif isinstance(item, str):
                    text_candidates.append(item)

                for candidate in text_candidates:
                    code = self._extract_code_from_text(candidate)
                    if code:
                        return code

        return ""

    def _evaluate_code_quality_for_coding(self, summary_data: Dict[str, Any]) -> None:
        """
        Evaluate code quality for coding environment and write it to summary data.

        Args:
            summary_data (Dict[str, Any]): Summary data written to JSONL.
        """
        if not isinstance(self.environment, CodingEnvironment):
            return

        summary_data["code_quality"] = {}
        code_path = os.path.join(self.environment.workspace_dir, "solution.py")
        code = self._read_code_from_file(code_path)
        if not code:
            extracted_code = self._extract_solution_from_iterations(summary_data)
            if extracted_code:
                os.makedirs(self.environment.workspace_dir, exist_ok=True)
                with open(code_path, "w", encoding="utf-8") as code_file:
                    code_file.write(extracted_code)
                code = extracted_code
                self.logger.info(
                    f"Recovered solution code from task results and wrote to {code_path}."
                )
            else:
                self.logger.warning(
                    f"No code found at {code_path}; skipped code quality evaluation."
                )
                return

        self.evaluator.evaluate_code_quality(task=self.task, code_result=code)
        summary_data["code_quality"] = self.evaluator.metrics.get("code_quality", {})
        self.logger.info(
            f"Code quality evaluation results: {summary_data['code_quality']}"
        )

    def __init__(self, config: Config):
        """
        Initialize the Engine with the given configuration.

        Args:
            config (Config): Configuration parameters.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.planning_method = config.engine_planner.get("planning_method", "naive")
        # Initialize Environment
        self.environment = self._initialize_environment(config.environment)
        # Initialize Agents
        self.agents = self._initialize_agents(config.agents)
        # Initialize AgentGraph
        self.graph = AgentGraph(self.agents, config)
        for agent in self.agents:
            agent.set_agent_graph(self.graph)
        # Initialize Memory
        self.memory = self._initialize_memory(config.memory)
        # Initialize Evaluator
        self.evaluator = Evaluator(metrics_config=config.metrics)
        self.task = config.task.get("content", "")
        self.output_format = config.task.get(
            "output_format",
            "You are free to define your own output format to answer the task properly.",
        )
        self.coordinate_mode = config.coordination_mode
        # Initialize EnginePlanner
        self.planner = EnginePlanner(
            agent_graph=self.graph,
            memory=self.memory,
            config=config.engine_planner,
            task=self.task,
            model=config.llm,
        )
        self.max_iterations = config.environment.get("max_iterations", 10)
        self.current_iteration = 0

        self.logger.info("Engine initialized.")

    def _initialize_environment(self, env_config: Dict[str, Any]) -> BaseEnvironment:
        """
        Initialize the environment based on configuration.

        Args:
            env_config (dict): Environment configuration.

        Returns:
            BaseEnvironment: An instance of the environment.

        Raises:
            ValueError: If the environment type is not supported.
        """
        env_type = env_config.get("type")

        if env_type == "Web":
            env1 = WebEnvironment(name="Web Environment", config=env_config)
            return env1
        elif env_type == "Base":
            env2 = BaseEnvironment(name="Base Environment", config=env_config)
            return env2
        elif env_type == "Research":
            env3 = ResearchEnvironment(name="Research Environment", config=env_config)
            return env3
        elif env_type == "Coding":
            env4 = CodingEnvironment(name="Coding Environment", config=env_config)
            return env4
        elif env_type == "WorldSimulation":
            env4 = WorldSimulationEnvironment(
                name="World Simulation Environment", config=env_config
            )
            return env4
        elif env_type == "Minecraft":
            env5 = MinecraftEnvironment(name="Minecraft Environment", config=env_config)
            return env5
        elif env_type == "DB":
            env6 = DBEnvironment(name="DB Environment", config=env_config)
            return env6
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")

    def _initialize_agents(
        self, agent_configs: List[Dict[str, Any]]
    ) -> List[BaseAgent]:
        """
        Initialize agents based on configurations.

        Args:
            agent_configs (List[dict]): List of agent configurations.

        Returns:
            List[BaseAgent]: List of agent instances.
        """
        agents = []
        llm = self.config.llm
        for agent_config in agent_configs:
            agent_llm = agent_config.get(
                "llm", llm
            )  # use agent-specific LLM if provided
            agent_type = agent_config.get("type")
            agent = BaseAgent(
                config=agent_config, env=self.environment, model=agent_llm
            )
            agents.append(agent)
            self.logger.debug(
                f"Agent '{agent.agent_id}' of type '{agent_type}' using LLM '{agent_llm}' initialized."
            )
            if isinstance(self.environment, MinecraftEnvironment):
                assert "agent_id" in agent_config and "agent_port" in agent_config
                self.environment.register_agent(
                    agent_config.get("agent_id"), agent_config.get("agent_port")
                )
            self.logger.debug(
                f"Agent '{agent.agent_id}' of type '{agent_type}' initialized."
            )
        return agents

    def _initialize_memory(
        self, memory_config: Dict[str, Any]
    ) -> Union[SharedMemory, BaseMemory]:
        """
        Initialize the shared memory mechanism.

        Args:
            memory_config (dict): Memory configuration.

        Returns:
            BaseMemory: An instance of the memory module.
        """
        memory_type = memory_config.get("type", "SharedMemory")
        memory: Union[BaseMemory, SharedMemory, None] = None
        if memory_type == "SharedMemory":
            memory = SharedMemory()
        else:
            memory = BaseMemory()
        self.logger.debug(f"Memory of type '{memory_type}' initialized.")
        return memory

    def graph_coordinate(self) -> None:
        """
        Graph-based coordination mode.
        """
        try:
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }
            # Initial assignment: Distribute the overall task to each agent
            self.logger.info("Initial task distribution to all agents.")
            initial_tasks = {
                agent.agent_id: self.task for agent in self.graph.get_all_agents()
            }
            agents_results = []

            # Initialize iteration_data for the initial assignment to match iterative structure
            iteration_data = {
                "iteration": self.current_iteration + 1,
                "task_assignments": {},
                "task_results": [],
                "summary": "",
                "continue_simulation": True,
                "communications": [],
            }
            communications = []
            for agent_id, task in initial_tasks.items():
                try:
                    agent = self.graph.get_agent(agent_id)
                    self.logger.info(f"Assigning initial task to {agent_id}: {task}")
                    # Assign the task to the agent
                    iteration_data_task_assignments = iteration_data.get(
                        "task_assignments"
                    )
                    assert isinstance(iteration_data_task_assignments, dict)
                    iteration_data_task_assignments[agent_id] = task
                    result, communication = agent.act(task)
                    self.logger.info(f"Processing result for agent '{agent.agent_id}'")
                    self.logger.info(f"Communication received: {communication}")
                    if communication:
                        self.logger.info(
                            f"Adding communication to list: {communication}"
                        )
                        communications.append(communication)
                    agents_results.append({agent_id: result})
                    # Record the result
                    task_result = {"agent_id": agent_id, "result": result}
                    iteration_data_task_results = iteration_data.get("task_results")
                    assert isinstance(iteration_data_task_results, list)
                    iteration_data_task_results.append(task_result)
                    self.logger.debug(
                        f"Agent '{agent_id}' completed initial task with result: {result}"
                    )
                except KeyError:
                    self.logger.error(f"Agent '{agent_id}' not found in the graph.")
                except Exception as e:
                    self.logger.error(
                        f"Error while executing initial task for agent '{agent_id}': {e}"
                    )
            iteration_data["communications"] = communications
            # Summarize outputs and update planner for the initial assignment
            summary = self._summarize_results(agents_results)
            self.logger.info(f"Initial Summary:\n{summary}")
            summary = self.planner.summarize_output(
                summary, self.task, self.output_format
            )
            iteration_data["summary"] = summary.content

            # Decide whether to continue or terminate after initial assignment
            if isinstance(self.environment, MinecraftEnvironment):
                try:
                    with open("../data/score.json", "r") as f:
                        block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                except:
                    block_hit_rate = 0.0
                self.logger.info(
                    f"Using a rule-based EnginePlanner. block_hit_rate is {block_hit_rate}"
                )
                continue_simulation = int(block_hit_rate) != 1
            else:
                continue_simulation = self.planner.decide_next_step(agents_results)
            iteration_data["continue_simulation"] = continue_simulation
            if not continue_simulation:
                self.logger.info(
                    "EnginePlanner decided to terminate the simulation after initial assignment."
                )
            else:
                self.planner.update_progress(summary)
                self.current_iteration += 1

            summary_data["iterations"].append(iteration_data)

            # Evaluate communication
            if iteration_data["communications"]:
                iteration_data_communications = iteration_data.get("communications")
                assert isinstance(iteration_data_communications, list)
                # communications_str = self._format_communications(iteration_data_communications)
                # self.evaluator.evaluate_communication(self.task, communications_str)
                self.evaluator.metrics["communication_score"].append(-1)
            else:
                self.logger.info("No communications to evaluate")
                # Store -1 if communications are empty
                self.evaluator.metrics["communication_score"].append(-1)

            # Evaluate planning
            # agent_profiles = self._get_agent_profiles()
            # iteration_data_task_assignments = iteration_data.get("task_assignments")
            # assert isinstance(iteration_data_task_assignments, dict)
            # agent_tasks_str = self._format_agent_tasks(iteration_data_task_assignments)
            # iteration_data_task_results = iteration_data.get("task_results")
            # assert isinstance(iteration_data_task_results, list)
            # results_str = self._format_results(iteration_data_task_results)
            # iteration_data_summary = iteration_data.get("summary")
            # assert isinstance(iteration_data_summary, str)
            # self.evaluator.evaluate_planning(iteration_data_summary, agent_profiles, agent_tasks_str, results_str)
            # self.evaluator.evaluate_kpi(self.task, results_str)
            self.evaluator.metrics["planning_score"].append(-1)

            end_on_iter_0 = False
            if not continue_simulation:
                end_on_iter_0 = True

            while self.current_iteration < self.max_iterations and not end_on_iter_0:
                iteration_data = {
                    "iteration": self.current_iteration + 1,
                    "task_assignments": {},
                    "task_results": [],
                    "summary": "",
                    "continue_simulation": True,
                    "communications": [],
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Starting iteration {self.current_iteration}")

                current_agents = self.graph.get_all_agents()
                current_tasks = {}
                agents_results = []
                communications = []

                for agent in current_agents:
                    try:
                        # Each agent plans its own task
                        task = agent.plan_task()
                        current_tasks[agent.agent_id] = task
                        iteration_data_task_assignments = iteration_data.get(
                            "task_assignments"
                        )
                        assert isinstance(iteration_data_task_assignments, dict)
                        iteration_data_task_assignments[agent.agent_id] = task
                        self.logger.info(
                            f"Agent '{agent.agent_id}' planned task: {task}"
                        )

                        # Agent acts on the planned task
                        result, communication = agent.act(task)
                        self.logger.info(
                            f"Processing result for agent '{agent.agent_id}'"
                        )
                        self.logger.info(f"Communication received: {communication}")
                        if communication:
                            self.logger.info(
                                f"Adding communication to list: {communication}"
                            )
                            communications.append(communication)
                        agents_results.append({agent.agent_id: result})
                        iteration_data_task_results = iteration_data.get("task_results")
                        assert isinstance(iteration_data_task_results, list)
                        iteration_data_task_results.append({agent.agent_id: result})
                        self.logger.debug(
                            f"Agent '{agent.agent_id}' executed task with result: {result}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error in agent '{agent.agent_id}' during planning or action: {e}"
                        )
                iteration_data["communications"] = communications
                # Summarize outputs and update planner
                summary = self._summarize_results(agents_results)
                self.logger.info(
                    f"Iteration {self.current_iteration} Summary:\n{summary}"
                )
                self.current_iteration += 1
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content

                # Evaluate communication
                if iteration_data["communications"]:
                    iteration_data_communications = iteration_data.get("communications")
                    assert isinstance(iteration_data_communications, list)
                    # communications_str = self._format_communications(iteration_data_communications)
                    # self.evaluator.evaluate_communication(self.task, communications_str)
                    self.evaluator.metrics["communication_score"].append(-1)
                else:
                    self.logger.info("No communications to evaluate")
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                # agent_profiles = self._get_agent_profiles()
                # iteration_data_task_assignments = iteration_data.get("task_assignments")
                # assert isinstance(iteration_data_task_assignments, dict)
                # agent_tasks_str = self._format_agent_tasks(iteration_data_task_assignments)
                # iteration_data_task_results = iteration_data.get("task_results")
                # assert isinstance(iteration_data_task_results, list)
                # results_str = self._format_results(iteration_data_task_results)
                # iteration_data_summary = iteration_data.get("summary")
                # assert isinstance(iteration_data_summary, str)
                # self.evaluator.evaluate_planning(iteration_data_summary, agent_profiles, agent_tasks_str, results_str)
                # self.evaluator.evaluate_kpi(self.task, results_str)
                self.evaluator.metrics["planning_score"].append(-1)
                # Decide whether to continue or terminate
                if isinstance(self.environment, MinecraftEnvironment):
                    try:
                        with open("../data/score.json", "r") as f:
                            block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                    except:
                        block_hit_rate = 0.0
                    self.logger.info(
                        f"Using a rule-based EnginePlanner. block_hit_rate is {block_hit_rate}"
                    )
                    continue_simulation = int(block_hit_rate) != 1
                else:
                    continue_simulation = self.planner.decide_next_step(agents_results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break

                # # Check if task is completed within the environment
                # if self.environment.is_task_completed():
                #     self.logger.info("Task has been completed successfully.")
                #     break
            # At the end, add the scores to summary_data

            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if isinstance(self.environment, CodingEnvironment):
                self._evaluate_code_quality_for_coding(summary_data)
                self.logger.info("Engine graph-based coordination loop completed.")
            # if self.environment.name == 'Research Environment':
            elif isinstance(self.environment, ResearchEnvironment):
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_task_research(self.task, iteration_data_summary)
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            elif isinstance(self.environment, MinecraftEnvironment):
                try:
                    with open("../data/score.json", "r") as f:
                        block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                except:
                    block_hit_rate = 0.0
                summary_data["task_evaluation"] = block_hit_rate * 5
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            self.logger.info("Engine graph-based coordination loop completed.")

        except Exception:
            self.logger.exception("An error occurred during graph-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Graph-based coordination simulation completed.")
            self._write_to_jsonl(summary_data)

    def star_coordinate(self) -> None:
        """
        Centralized coordination mode.
        """
        try:
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
                "final_output": "",
            }
            agents_results: List[Dict[str, Any]] = []
            while self.current_iteration < self.max_iterations:
                iteration_data: Dict[str, Any] = {
                    "iteration": self.current_iteration + 1,
                    "task_assignments": {},
                    "task_results": [],
                    "summary": "",
                    "continue_simulation": True,
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Starting iteration {self.current_iteration}")

                # Assign tasks to agents
                assignment = self.planner.assign_tasks(
                    planning_method=self.planning_method
                )
                tasks = assignment.get("tasks", {})
                iteration_data["task_assignments"] = tasks
                self.logger.info(f"Assigned tasks: {tasks}")

                # Assign tasks to agents
                agents_results = []
                communications = []
                for agent_id, task in tasks.items():
                    try:
                        agent = self.graph.get_agent(agent_id)
                        self.logger.info(f"Assigning task to {agent_id}: {task}")
                        result, communication = agent.act(task)
                        agents_results.append({agent_id: result})
                        if communication:
                            communications.append(communication)

                        self.logger.debug(
                            f"Agent '{agent_id}' completed task with result: {result}"
                        )
                    except KeyError:
                        self.logger.error(f"Agent '{agent_id}' not found in the graph.")
                    except Exception as e:
                        self.logger.error(
                            f"Error while executing task for agent '{agent_id}': {e}"
                        )
                iteration_data["task_results"] = agents_results
                iteration_data["communications"] = communications
                # Update progress based on agents' results
                summary = self._summarize_results(agents_results)
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content
                self.logger.info(summary)
                self.planner.update_progress(summary)
                self.current_iteration += 1

                # Evaluate communication
                if iteration_data["communications"]:
                    communications_str = self._format_communications(
                        iteration_data["communications"]
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data["task_assignments"]
                )
                results_str = self._format_results(iteration_data["task_results"])
                self.evaluator.evaluate_planning(
                    iteration_data["summary"],
                    agent_profiles,
                    agent_tasks_str,
                    results_str,
                )
                self.evaluator.evaluate_kpi(self.task, results_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(agents_results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break

                if self.current_iteration >= self.max_iterations:
                    self.logger.info("Maximum iterations reached.")
                    break
            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            if isinstance(self.environment, CodingEnvironment):
                self._evaluate_code_quality_for_coding(summary_data)
                self.logger.info("Engine star-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine star-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine star-based coordination loop completed.")
            self.logger.info("Engine simulation loop completed.")

        except Exception:
            self.logger.exception("An error occurred during simulation.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Simulation completed.")
            self._write_to_jsonl(summary_data)

    def chain_coordinate(self) -> None:
        """
        Chain-based coordination mode.
        """
        try:
            self.logger.info("Starting chain-based coordination.")
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }
            # Start with the initial agent
            current_agent = self._select_initial_agent()
            if not current_agent:
                self.logger.error("No initial agent found for chain.")
                return

            max_chain_length = self.max_iterations * len(
                self.agents
            )  # Or define a separate chain length limit
            chain_length = 0

            task = self.task
            agents_results = []

            while current_agent and chain_length < max_chain_length:
                iteration_data = {
                    "chain_length": chain_length + 1,
                    "current_agent": current_agent.agent_id,
                    "result": None,
                    "continue_simulation": True,
                    "task_assignments": {},
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Agent '{current_agent.agent_id}' is executing task.")
                result, communication = current_agent.act(task)
                result_str = f"AgentID: '{current_agent.agent_id}' completed task with result: {result}"
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                iteration_data_task_assignments[current_agent.agent_id] = task
                agents_results.append({current_agent.agent_id: result})
                iteration_data["result"] = result
                self.logger.info(
                    f"Agent '{current_agent.agent_id}' completed task with result: {result}"
                )
                # Get profiles of other agents
                agent_profiles = self.graph.get_agent_profiles_linked(
                    current_agent.agent_id
                )
                # Current agent chooses the next agent
                next_agent_id, plan = current_agent.plan_next_agent(
                    result, agent_profiles
                )
                current_agent_ = current_agent
                try:
                    current_agent = self.graph.get_agent(next_agent_id)
                except Exception:
                    self.logger.error(
                        f"Agent '{next_agent_id}' not found in the graph. keep the same agent."
                    )
                    current_agent = current_agent_
                task = plan
                chain_length += 1
                self.planner.update_progress(result)
                iteration_data["communications"] = communication

                # Evaluate communication
                if iteration_data["communications"]:
                    iteration_data_communications = iteration_data.get("communications")
                    assert isinstance(iteration_data_communications, list)
                    communications_str = self._format_communications(
                        iteration_data_communications
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                summary = self._summarize_results(agents_results)
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content

                # Evaluate planning
                agent_profiles_self = self._get_agent_profiles()
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data_task_assignments
                )
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_planning(
                    iteration_data_summary, agent_profiles_self, agent_tasks_str, result
                )
                self.evaluator.evaluate_kpi(self.task, result_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(
                    [{"root_agent": result}]
                )
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break
            # Update progress
            summary = self._summarize_results(agents_results)
            self.logger.info(f"Chain execution Summary:\n{summary}")
            self.planner.update_progress(summary)

            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                # summary_data['task_evaluation'] = self.evaluator.metrics["task_evaluation"]
                self.logger.info("Engine chain-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine chain-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine chain-based coordination loop completed.")
            self.logger.info("Chain-based coordination simulation completed.")

        except Exception:
            self.logger.exception("An error occurred during chain-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Chain-based coordination simulation completed.")
            summary_data["token_usage"] = self._get_totoal_token_usage()
            self._write_to_jsonl(summary_data)

    def tree_coordinate(self) -> None:
        """
        Tree-based coordination mode.
        """
        try:
            self.logger.info("Starting tree-based coordination.")
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }

            root_agent = self.graph.get_root_agent()
            if not root_agent:
                self.logger.error("No root agent found in the tree.")
                return
            # Start the coordination from the root agent
            while self.current_iteration < self.max_iterations:
                iteration_data: Dict[str, Any] = {
                    "iteration": self.current_iteration + 1,
                    "root_agent": root_agent.agent_id,
                    "result": None,
                    "continue_simulation": True,
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.current_iteration += 1
                self.logger.info(f"Starting iteration {self.current_iteration}")
                results, communication, tasks = self._execute_agent_task_recursive(
                    root_agent, self.task
                )
                # Update progress
                summary = self._summarize_results(results)
                summary = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary.content
                self.logger.info(
                    f"Iteration {self.current_iteration} Summary:\n{summary}"
                )
                self.planner.update_progress(summary)
                iteration_data["communications"] = communication
                iteration_data["task_assignments"] = tasks
                iteration_data["task_results"] = results
                # Evaluate communication
                if iteration_data["communications"]:
                    communications_str = self._format_communications(
                        iteration_data["communications"]
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data["task_assignments"]
                )
                results_str = self._format_results(iteration_data["task_results"])
                self.evaluator.evaluate_planning(
                    iteration_data["summary"],
                    agent_profiles,
                    agent_tasks_str,
                    results_str,
                )
                self.evaluator.evaluate_kpi(self.task, results_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break
            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            if isinstance(self.environment, CodingEnvironment):
                self._evaluate_code_quality_for_coding(summary_data)
                self.logger.info("Engine tree-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine tree-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine tree-based coordination loop completed.")
            self.logger.info("Tree-based coordination simulation completed.")

        except Exception:
            self.logger.exception("An error occurred during tree-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Tree-based coordination simulation completed.")
            self._write_to_jsonl(summary_data)

    def _execute_agent_task_recursive(self, agent: BaseAgent, task: str) -> Any:
        """
        Recursively execute tasks starting from the given agent.

        Args:
            agent (BaseAgent): The agent to execute task.
            task (str): The task to execute.

        Returns:
            Any: The result of the agent's execution.
        """
        self.logger.info(f"Agent '{agent.agent_id}' is executing task.")
        tasks = []
        print(agent.children)
        if len(agent.children) > 0:
            print("******************start recursive******************")
            # Agent assigns tasks to children
            tasks_for_children = agent.plan_tasks_for_children(task)
            tasks.append(tasks_for_children)
            children_results = []
            communications = []
            for child in agent.children:
                child_task = tasks_for_children.get(child.agent_id, "")
                if child_task:
                    (
                        child_result,
                        communication,
                        tasks_,
                    ) = self._execute_agent_task_recursive(child, child_task)
                    tasks += tasks_
                    if communication:
                        communications.append(communication)
                    children_results += child_result
            # Agent may also act itself
            results_str = "\n".join(
                json.dumps(result)[:500] for result in children_results
            )

            task_for_father = (
                task
                + "\nHere are the results of the children: "
                + results_str
                + "\nPlease don't repeat the same task and continue to work on the original task. You may also need to communicate with other agents or summarize the results or just continue to work on the original task."
            )
            own_result, communication = agent.act(task_for_father)

            if communication:
                communications.append(communication)
            communications_str = "\n".join(communications) if communications else None
            # # Combine results
            # combined_result = agent.summarize_results(children_results, own_result)
            results = [
                {"agent_id": agent.agent_id, "result": own_result}
            ] + children_results
            return results, communications_str, tasks
        else:
            # Agent directly acts on the task
            result, communication = agent.act(task)
            return (
                [{"agent_id": agent.agent_id, "result": result}],
                communication,
                tasks,
            )

    def _select_initial_agent(self) -> Optional[BaseAgent]:
        """
        Select the initial agent to start the chain.

        Returns:
            Optional[BaseAgent]: The initial agent, or None if not found.
        """
        # For simplicity, select an agent based on some criteria.
        # Here, we'll select the agent with the highest priority or a predefined agent.
        # Alternatively, we could prompt the LLM to select the starting agent.

        # Example: Select agent1 as the starting agent
        starting_agent_id = "agent1"
        if starting_agent_id in [agent.agent_id for agent in self.agents]:
            return self.graph.get_agent(starting_agent_id)
        else:
            self.logger.error(f"Starting agent '{starting_agent_id}' not found.")
            return None

    def start(self) -> None:
        """
        Start the engine to run the simulation.
        """
        self.logger.info("Engine starting simulation.")
        if isinstance(self.environment, MinecraftEnvironment):
            self.environment.launch()
        if self.coordinate_mode == "star":
            self.logger.info("Running in centralized coordination mode.")
            self.star_coordinate()
        elif self.coordinate_mode == "graph":
            self.logger.info("Running in graph-based coordination mode.")
            self.graph_coordinate()
        elif self.coordinate_mode == "chain":
            self.logger.info("Running in chain-based coordination mode.")
            self.chain_coordinate()
        elif self.coordinate_mode == "tree":
            self.logger.info("Running in tree-based coordination mode.")
            self.tree_coordinate()
        else:
            self.logger.error(f"Unsupported coordinate mode: {self.coordinate_mode}")
            raise ValueError(f"Unsupported coordinate mode: {self.coordinate_mode}")
        if isinstance(self.environment, MinecraftEnvironment):
            self.environment.finish()

    def _should_terminate(self) -> bool:
        """
        Determine whether the simulation should terminate.

        Returns:
            bool: True if should terminate, False otherwise.
        """
        # Placeholder for any additional termination conditions
        return False

    def _summarize_results(self, agents_results: List[Dict[str, Any]]) -> str:
        """
        Summarize the agents' results into a string.

        Args:
            agents_results (Dict[str, Any]): The results from all agents.

        Returns:
            str: The summary string.
        """
        summary = "Agents' Results Summary:\n"
        # for agent_id, result in agents_results.items():
        #     summary += f"- {agent_id}: {result}\n"
        for result in agents_results:
            shorten_result = f"- {result}"
            shorten_result = shorten_result[:1000]
            summary += f"{shorten_result}\n"

        self.logger.debug(f"Summarized agents' results:\n{summary}")
        return summary

    def _write_to_jsonl(self, summary_data: Dict[str, Any]) -> None:
        """
        Write summary data to the JSONL file.

        Args:
            summary_data (List[Dict[str, Any]]): Summary data to write to the JSONL file.
        """
        file_path = self.config.output.get(
            "file_path", "result/discussion_output.jsonl"
        )
        try:
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(file_path, "a") as jsonl_file:
                print(summary_data)
                jsonl_file.write(json.dumps(summary_data) + "\n")

                jsonl_file.flush()
            self.logger.info(f"Summary data successfully written to {file_path}")
        except IOError as e:
            self.logger.error(f"Failed to write summary data to {file_path}: {e}")

    def _get_final_ooutput_in_graph(self):
        """
        Get the final output graph.

        Returns:
            Dict[str, Any]: The final output graph.
        """
        return self.graph.get_output_graph()

    def _format_communications(self, communications: List[Any]) -> str:
        """
        Formats the communications list into a string suitable for evaluator input.
        """
        # Assuming each communication is a string or can be converted to string
        return "\n".join(str(c) for c in communications)

    def _get_agent_profiles(self) -> str:
        """
        Retrieves and formats agent profiles into a string.
        """
        agent_profiles = []
        for agent in self.graph.get_all_agents():
            # Assuming agent has attributes agent_id and profile
            agent_profiles.append(
                f"Agent ID: {agent.agent_id}, Profile: {agent.profile}"
            )
        return "\n".join(agent_profiles)

    def _format_agent_tasks(self, agent_tasks: Dict[str, Any]) -> str:
        """
        Formats agent tasks into a string.
        """
        try:
            return "\n".join(
                f"Agent {agent_id}: Task: {task}"
                for agent_id, task in agent_tasks.items()
            )
        except Exception:
            return "\n".join(json.dumps(item) for item in agent_tasks)

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Formats results into a string.
        """
        results_str = []
        for result in results:
            if "agent_id" in result and "result" in result:
                agent_id = result["agent_id"]
                res_content = result["result"]
                results_str.append(f"AgentID: {agent_id}: Result: {res_content}")
            else:
                for agent_id, res_content in result.items():
                    results_str.append(f"Agent {agent_id}: Result: {res_content}")
        return "\n".join(results_str)

    def _get_totoal_token_usage(self) -> int:
        """
        Get the total token usage by all agents.
        """
        return (
            sum(agent.token_usage for agent in self.graph.get_all_agents())
            + self.planner.token_usage
        )
