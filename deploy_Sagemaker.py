import sagemaker
from sagemaker.model import Model

# Initialize SageMaker session for deleting endpoint
sagemaker_session = sagemaker.Session()

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

# Attempt to delete existing endpoint and config to avoid caching issues
endpoint_name = "parler-tts-endpoint"
try:
    sagemaker_session.delete_endpoint(endpoint_name=endpoint_name)
    print(f"Deleted existing endpoint: {endpoint_name}")
    # Wait for endpoint to be fully deleted if necessary, or handle EndpointNotFound
    waiter = sagemaker_session.sagemaker_client.get_waiter('endpoint_deleted')
    waiter.wait(EndpointName=endpoint_name)
except Exception as e:
    if "Could not find endpoint" in str(e) or "EndpointNotFound" in str(e):
        print(f"Endpoint {endpoint_name} not found, proceeding.")
    else:
        print(f"Error deleting endpoint {endpoint_name}: {e}")

try:
    sagemaker_session.delete_endpoint_config(endpoint_config_name=endpoint_name)
    print(f"Deleted existing endpoint config: {endpoint_name}")
except Exception as e:
    if "Could not find endpoint config" in str(e):
        print(f"Endpoint config {endpoint_name} not found, proceeding.")
    else:
        print(f"Error deleting endpoint config {endpoint_name}: {e}")

# Create a SageMaker Model object for a custom container
model = Model(
    image_uri="534437858001.dkr.ecr.ap-south-1.amazonaws.com/parler-tts-5:latest",
    role=role,
    sagemaker_program=None  # Explicitly tell SageMaker this is a custom CMD
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="g6.2xlarge",  # Using GPU instance for better performance
    endpoint_name=endpoint_name
)

print(f"Endpoint deployed successfully: {predictor.endpoint_name}") 