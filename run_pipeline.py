from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
import os

ws = Workspace.from_config()
compute_target = ws.compute_targets["eVTOL-Compute"]

env = Environment.from_conda_specification(name="eVTOL-env", file_path="../requirements.txt")

model_data = PipelineData("model_data", datastore=ws.get_default_datastore())

train_step = PythonScriptStep(
    name="Train_Model",
    script_name="../ai_model/train_model.py",
    arguments=["--output_model", model_data],
    outputs=[model_data],
    compute_target=compute_target,
    source_directory=os.path.dirname(os.path.abspath(__file__)),
)

eval_step = PythonScriptStep(
    name="Evaluate_Model",
    script_name="../ai_model/evaluate_model.py",
    arguments=["--input_model", model_data],
    inputs=[model_data],
    compute_target=compute_target,
    source_directory=os.path.dirname(os.path.abspath(__file__)),
)

simulate_step = PythonScriptStep(
    name="Simulate_Autonomous_Flight",
    script_name="../main.py",
    compute_target=compute_target,
    source_directory=os.path.dirname(os.path.abspath(__file__)),
)

pipeline = Pipeline(workspace=ws, steps=[train_step, eval_step, simulate_step])

experiment = Experiment(workspace=ws, name="eVTOL-AI-FullPipeline")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)
