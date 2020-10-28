import os
UPLOAD_FLODER = 'static/uploads/'
DETECTED_FOLDER ='static/predict/'

def remove_file(path):

	# removing the file
	if not os.remove(path):

		# success message
		print(f"{path} is removed successfully")

	else:

		# failure message
		print(f"Unable to delete the {path}")


for root_folder, folders, files in os.walk(UPLOAD_FLODER):
    for file in files:
        file_path = os.path.join(UPLOAD_FLODER, file)
        remove_file(file_path)

for root_folder, folders, files in os.walk(DETECTED_FOLDER):
    for file in files:
        file_path = os.path.join(DETECTED_FOLDER, file)
        remove_file(file_path)
