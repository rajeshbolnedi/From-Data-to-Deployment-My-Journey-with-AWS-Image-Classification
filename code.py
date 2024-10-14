#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sagemaker import image_uris, TrainingInput
import sagemaker

# region and SageMaker role
region = 'us-east-2'
role = 'arn:aws:iam::982534360445:role/service-role/AmazonSageMaker-ExecutionRole-20241010T175777'

# Retrieve the image URI for the image classification algorithm
training_image = image_uris.retrieve(framework='image-classification', region=region)

s3_bucket = 'raj-imagestorage'
output_path = f's3://{s3_bucket}/output'

image_classifier = sagemaker.estimator.Estimator(
    image_uri=training_image,
    role=role,
    instance_count=1,
    instance_type='ml.p2.xlarge',
    output_path=output_path,
    sagemaker_session=sagemaker.Session()
)

# number of training samples and set mini_batch_size accordingly
num_training_samples = 24
mini_batch_size = 4

# Set hyperparameters
image_classifier.set_hyperparameters(
    num_layers=18,
    use_pretrained_model=1,
    num_classes=2,
    mini_batch_size=mini_batch_size,
    num_training_samples=num_training_samples
)

#  S3 paths for training and validation datasets using TrainingInput
train_data = TrainingInput(
    s3_data=f's3://{s3_bucket}/train',
    content_type='application/x-image'
)

train_lst_data = TrainingInput(
    s3_data=f's3://{s3_bucket}/train/train.lst',
    content_type='application/x-image'
)

validation_data = TrainingInput(
    s3_data=f's3://{s3_bucket}/validation',
    content_type='application/x-image'
)

validation_lst_data = TrainingInput(
    s3_data=f's3://{s3_bucket}/validation/validation.lst',
    content_type='application/x-image'
)

# Fit the model with both train, train_lst, validation, and validation_lst channels
image_classifier.fit(
    {
        'train': train_data,
        'train_lst': train_lst_data,
        'validation': validation_data,
        'validation_lst': validation_lst_data
    }
)


# In[32]:


# Deploying the model
predictor = image_classifier.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'  # Choose an instance type that fits your workload
)


# In[35]:


import boto3
from PIL import Image
from io import BytesIO
import numpy as np

# Initialize S3 client and specify bucket and object key
s3 = boto3.client('s3')
bucket_name = 'raj-imagestorage'
object_key = 'test/test1.jpeg'  # Update with your correct key

# Function to download and preprocess the image
def download_and_preprocess_image(bucket_name, object_key):
    s3_response = s3.get_object(Bucket=bucket_name, Key=object_key)
    image_data = s3_response['Body'].read()
    
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))
    
    # Convert image to RGB in case it's not, and save as JPEG
    image = image.convert('RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'JPEG')
    byte_io.seek(0)
    
    return byte_io.read()

# Preprocess the image and get it as bytes in JPEG format
processed_image_bytes = download_and_preprocess_image(bucket_name, object_key)

# Make prediction using 'application/x-image' content type
response = predictor.predict(processed_image_bytes, initial_args={'ContentType': 'application/x-image'})
print(response)


# In[39]:


# testing the m

import numpy as np
import json

# Assuming `response` is the output from the model, e.g., b'[0.0, 1.0]'
# Decode the response and load it as a list
decoded_response = response.decode('utf-8')
response_array = json.loads(decoded_response)

# Define the mapping from indices to class labels
class_labels = {0: 'cat', 1: 'dog'}

# Find the index of the maximum value in the response (predicted class)
predicted_class_index = np.argmax(response_array)

# Get the class label for the predicted index
predicted_label = class_labels[predicted_class_index]

print(f"The model predicts this image is a: {predicted_label}")



# In[40]:


predictor.delete_endpoint()


# In[27]:


#downloading the model artifacts.
from sagemaker.s3 import S3Downloader

S3Downloader.download(
    s3_uri=output_path, 
    local_path='artifacts_image'
)


# In[ ]:




