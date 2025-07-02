import boto3
import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ModelStep
from sagemaker.processing import ScriptProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

region = boto3.Session().region_name
role = sagemaker.get_execution_role()
session = sagemaker.Session()

bucket = session.default_bucket()

# Parameters
input_data = ParameterString(name="InputDataUrl", default_value=f"s3://{bucket}/mlops-churn-processed-data/preprocessed.csv")
model_package_group_name = "ChurnModelPackageGroup"
pipeline_name = "churn-pipeline"

# Processing Step - Run preprocessing.py
script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region),
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=script_processor,
    inputs=[sagemaker.processing.ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[sagemaker.processing.ProcessingOutput(output_name="processed_data", source="/opt/ml/processing/output")],
    code="preprocessing.py"
)

# Training Step
xgb_container = sagemaker.image_uris.retrieve("xgboost", region, "1.6-1")

xgb_estimator = XGBoost(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.6-1",
    py_version="py3",
    output_path=f"s3://{bucket}/mlops-churn-model-artifacts",
    hyperparameters={"objective": "binary:logistic", "eval_metric": "logloss", "use_label_encoder": False}
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={"train": processing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri}
)

# Model Registration Step
model = Model(
    image_uri=xgb_container,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=session,
)

register_model_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
    )
)

# Build pipeline
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[input_data],
    steps=[processing_step, train_step, register_model_step],
    sagemaker_session=session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()
