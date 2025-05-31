import sagemaker
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve import Mode
from inference import ParlerTTSInferenceSpec

# Initialize SageMaker session
session = sagemaker.Session()

# Use a specific role ARN
role = "arn:aws:iam::534437858001:role/service-role/AmazonSageMaker-ExecutionRole-20250226T082807"

# Sample input and output for schema
sample_input = {
    "prompt": "Hello, how are you?",
    "description": "A female speaker with a clear voice"
}
sample_output = {
    "audio_base64": "base64_encoded_audio_string"
}

# Create inference spec
inference_spec = ParlerTTSInferenceSpec()

# Create model builder
model_builder = ModelBuilder(
    mode=Mode.SAGEMAKER_ENDPOINT,
    model_server=None,  # Using custom container
    schema_builder=SchemaBuilder(sample_input, sample_output),
    inference_spec=inference_spec,
    role_arn=role,
    image_uri="534437858001.dkr.ecr.ap-south-1.amazonaws.com/parler-tts-5:latest"
)

# Build the model
model = model_builder.build()

# Deploy the model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",  # Using GPU instance for better performance
    endpoint_name="parler-tts-endpoint"
)

print(f"Endpoint deployed successfully: {predictor.endpoint_name}") 