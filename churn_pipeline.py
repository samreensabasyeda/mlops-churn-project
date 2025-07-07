import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.xgboost.estimator import XGBoost
import sagemaker

# Environment setup
region = boto3.Session().region_name
session = PipelineSession()
role = "arn:aws:iam::911167906047:role/SageMakerChurnRole"
bucket = "mlops-churn-model-artifacts"

# Pipeline parameter
input_data = ParameterString(
    name="InputDataUrl",
    default_value="s3://mlops-churn-processed-data/preprocessed.csv"
)

# Preprocessing Step
script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    command=["python3"],
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    sagemaker_session=session
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=script_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/processed/train"
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination=f"s3://{bucket}/processed/validation"
        )
    ],
    code="preprocessing.py"
)

# Training Step
xgb_container = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")

xgb_estimator = XGBoost(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.5-1",
    py_version="py3",
    output_path=f"s3://{bucket}/output",
    sagemaker_session=session,
    hyperparameters={
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "seed": 42
    },
    environment={
        "MLFLOW_TRACKING_URI": "http://13.203.193.28:30172/",
        "MLFLOW_EXPERIMENT_NAME": "ChurnPrediction"
    }
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Model Registration Step
model = Model(
    image_uri=xgb_container,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=session,
    env={
        "MLFLOW_TRACKING_URI": "http://13.203.193.28:30172/",
        "MLFLOW_EXPERIMENT_NAME": "ChurnPrediction"
    }
)

register_model_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name="ChurnModelPackageGroup",
        approval_status="PendingManualApproval"
    )
)

# Final Pipeline
pipeline = Pipeline(
    name="churn-pipeline",
    parameters=[input_data],
    steps=[processing_step, train_step, register_model_step],
    sagemaker_session=session
)

# Execute pipeline
if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()