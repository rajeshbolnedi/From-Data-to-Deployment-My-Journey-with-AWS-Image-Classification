import json
import boto3

s3 = boto3.client('s3')
runtime = boto3.client('sagemaker-runtime')

# Define a mapping for class labels
class_labels = {0: 'cat', 1: 'dog'}

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    image_key = event['Records'][0]['s3']['object']['key']

    # Load image from S3
    s3_response = s3.get_object(Bucket=bucket, Key=image_key)
    image_bytes = s3_response['Body'].read()

    # Send image to SageMaker endpoint for classification
    response = runtime.invoke_endpoint(
        EndpointName='image-classification-2024-10-13-21-46-17-001',
        ContentType='application/x-image',
        Body=image_bytes
    )
    
    # Parse the response
    result = json.loads(response['Body'].read().decode())

    # Determine the predicted label
    predicted_class_index = result.index(max(result))  # Get the index of the highest confidence score
    predicted_label = class_labels[predicted_class_index]

    # Log and return the result
    print("Inference result:", result)
    print("Predicted label:", predicted_label)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': predicted_label})
    }
