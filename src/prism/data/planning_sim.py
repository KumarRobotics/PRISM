from typing import Any, Dict

from spine.spine import SPINE

from prism.data.graph_sim import GraphSim


class PlanningSim:
    def __init__(self, debug=True):
        self.debug = debug

    def query_planner(self, llm_input: str, planner: SPINE) -> None:
        resp, success, logs = planner.request(llm_input)

        if success:
            if self.debug:
                print(f"success: {success}")

                print("--feedback--\n")
                for log in logs:
                    print(log)
                print("\n--")

            # pprint.PrettyPrinter().pprint(resp)

            plan = resp["plan"]
            reason = resp["reasoning"]

            if self.debug:
                print(f"plan:")
                for action, arg in plan:
                    parsed_arg = arg
                    print(f"\t{action}( {parsed_arg} )")

                print(f"reason: {reason}")

            return resp
        else:
            print(f"failed")
            print(resp)
            return {}

    def run_planning(
        self,
        *,
        llm_planner: SPINE,
        task: str,
        graph_data_gen: GraphSim,
        max_iterations=10,
    ) -> Dict[str, Any]:
        done = False
        planner_input = f"task: {task}"

        for _ in range(max_iterations):
            out = self.query_planner(planner_input, llm_planner)

            # if plan is badly formed just return
            if "plan" not in out:
                return {"response": out}

            for step in out["plan"]:
                function, arg = step

                if function == "answer":
                    done = True

                have_updates = graph_data_gen.take_action(function, arg)

                if have_updates:
                    break

            if done:
                break

            planner_input = graph_data_gen.get_updator().form_updates()

        return out
