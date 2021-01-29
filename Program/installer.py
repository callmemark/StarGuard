import os


def install_dependecies():
	print("install will begin...")
	try:
		os.system("pip install numpy")
		os.system("pip install pip install sklearn")
		os.system("pip install matplotlib")
		os.system("pip install scikit-image")
		os.system("pip install opencv-python")
		os.system("pip install pandas")
		os.system("pip install Pillow")

	except Exception as error:
		print("error while downloading: " + str(error))


install_dependecies()