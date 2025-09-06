from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


connection_string=os.getenv('connection_string')
container_name=os.getenv("container_name")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Upload single file
local_file = "/Users/varunnegi/assignments/66dec1c5c91ead1b5aa75de0-2.pdf"
with open(local_file, "rb") as data:
    container_client.upload_blob(name="files/66dec1c5c91ead1b5aa75de0-2.pdf", data=data, overwrite=True)

print("Upload successful!")

# # Delete all blobs
# for blob in container_client.list_blobs():
#     container_client.delete_blob(blob.name)
#     print(f"Deleted: {blob.name}")

# print("All files deleted.")
