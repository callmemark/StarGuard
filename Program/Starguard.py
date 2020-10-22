try:
	import matplotlib.image as mpimg
	import numpy as np
	import pandas as pd
	from PIL import Image
	from skimage.feature import hog
	from skimage.io import imread
	from skimage import data, exposure
	from astropy.io import fits
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.model_selection import train_test_split
	from sklearn import metrics
	import cv2 as cv
	import matplotlib.pyplot as plt
	import time
	import os
	
except Exception as error:
	print("Library import error: " + str(error))
	print("if you think this is a bug please open a new issue in project repository")



class StarGuard():
	def __init__(self, image_size = 200, K = 3):
		self.image_size = image_size
		self.data_loop_count = 20
		self.neighbors = K



	def knnAlgorithmDetector(self, dataset, frame):
		X = dataset.iloc[:, :-1].values
		y = dataset["classification"]
		
		knn = KNeighborsClassifier(n_neighbors = self.neighbors)
		knn.fit(X, y)
		prediction = knn.predict(frame)
		return prediction



	def testDatasetAccurracy(self):
		X = self.dataset.iloc[:, :-1].values
		y = self.dataset["classification"]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		knn = KNeighborsClassifier(n_neighbors = self.neighbors)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)

		print(y_pred)
		print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



	def showDataset(self, dataset):
		print("loading dataset")
		loaded_dataset = pd.read_csv(dataset)
		print(loaded_dataset)



	def encodeHog(self, image_input):
	    image = imread(image_input)
	    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(2, 2), 
	    	visualize = True, multichannel = True)

	    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
	    return hog_image_rescaled
		    


	def toOneDimArray(self, multi_dimensional_array):
		flattend_array = multi_dimensional_array.flatten()
		return flattend_array



	def createImageHogDataset(self, dataset_name, image):
		try:
			resized_image = Image.open(image)
			resized_image = resized_image.resize((self.image_size,self.image_size),Image.ANTIALIAS)
			resized_image.save("deleteThisAfterDatasetIsCreated.jpg",optimize=True,quality=100)
			image = "deleteThisAfterDatasetIsCreated.jpg"
			hog_encoded_image = self.encodeHog(image)

			image_hog_data = np.array(hog_encoded_image).reshape(1, self.image_size * self.image_size)
			dataset_array = np.array([])
			loop_count = self.data_loop_count

			while loop_count > 0:
				dataset_array = np.append(dataset_array, image_hog_data)
				loop_count -= 1
				print("looped" + str(loop_count))

				if loop_count == 0:
					break

			print(dataset_array.shape)

			dataset_array = dataset_array.reshape(self.data_loop_count, self.image_size * self.image_size)
			print("shape" + str(dataset_array.shape))
			dataset = pd.DataFrame(dataset_array, columns = map(str, range(self.image_size * self.image_size))) 
			dataset["classification"] = "normal"

			dataset.to_csv(dataset_name, index = False)
			print("newdataset added")
			
		except Exception as error:
			print("Error:" +  str(error))



	def addDataToImageHogDataset(self, dataset_name, image, normal):
		try:
			resized_image = Image.open(image)
			resized_image = resized_image.resize((self.image_size,self.image_size),Image.ANTIALIAS)
			resized_image.save("deleteThisAfterDatasetIsCreated.jpg",optimize=True,quality=100)
			image = "deleteThisAfterDatasetIsCreated.jpg"
			hog_encoded_image = self.encodeHog(image)

			image_hog_data = np.array(hog_encoded_image).reshape(1, self.image_size * self.image_size)
			dataset_array = np.array([])
			loop_count = self.data_loop_count

			while loop_count > 0:
				dataset_array = np.append(dataset_array, image_hog_data)
				loop_count -= 1
				print("looped" + str(loop_count))

				if loop_count == 0:
					break

			print(dataset_array.shape)

			dataset_array = dataset_array.reshape(self.data_loop_count, self.image_size * self.image_size)

			print("shape" + str(dataset_array.shape))
			dataset = pd.DataFrame(dataset_array, columns = map(str, range(self.image_size * self.image_size))) 
			
			if normal == True:
				dataset["classification"] = "normal"
			elif normal == False:
				dataset["classification"] = "anomaly"
			else:
				print("runtime error")

			dataset.to_csv(dataset_name, index = False, header=None, mode='a')
			print("newdataset added")

		except Exception as error:
			print(str(error))



	def createDatasetFromFits(self, dataset_name,fits_image):
		try:
			fits_image = fits_image
			fits_data, header = fits.getdata(fits_image, ext=0, header = True)

			fits_array = np.array(fits_data[0])
			array_shape = fits_array.shape

			array_value_count = array_shape[0]

			fits_array.reshape(1, array_value_count)

			loop_count = self.data_loop_count
			fits_dataset = np.array([])
			while loop_count > 0:
				fits_dataset = np.append(fits_dataset, fits_array)
				loop_count -= 1
				print("looped" + str(loop_count))

				if loop_count == 0:
					break

			print(fits_dataset.shape)

			fits_dataset = fits_dataset.reshape(self.data_loop_count, int(array_value_count / 2) * 2)

			print("shape" + str(fits_dataset.shape))
			dataset = pd.DataFrame(fits_dataset, columns = map(str, range(int(array_value_count / 2) * 2)))
			dataset["classification"] = "normal"

			dataset.to_csv(dataset_name, index = False)
			print("newdataset added")


		except Exception as error:
			print("Error:" + str(error))



	def addDataToFitsDataSet(self, target_dataset, fits_files, normal, count = 20):
		fits_lits = np.array(fits_files)
		new_data_dimension = fits_lits.ndim
		if isinstance(normal, bool) and new_data_dimension == 1:
			try:
				list_count = fits_lits.shape[0]

				for fits_image in fits_files:
					current_file = fits_image
					fits_data, header = fits.getdata(current_file, ext=0, header = True)

					fits_array = np.array(fits_data[0])
					array_shape = fits_array.shape

					array_value_count = array_shape[0]

					fits_array.reshape(1, array_value_count)

					loop_count = self.data_loop_count
					fits_dataset = np.array([])
					while loop_count > 0:
						fits_dataset = np.append(fits_dataset, fits_array)
						loop_count -= 1
						print("looped" + str(loop_count))

						if loop_count == 0:
							break

					print(fits_dataset.shape)
					fits_dataset = fits_dataset.reshape(self.data_loop_count, int(array_value_count / 2) * 2)

					print("shape" + str(fits_dataset.shape))
					dataset = pd.DataFrame(fits_dataset, columns = map(str, range(int(array_value_count / 2) * 2)))

					if normal == True:
						dataset["classification"] = "normal"
					elif normal == False:
						dataset["classification"] = "anomaly"
					else:
						print("runtime error")

					dataset.to_csv(target_dataset, index = False, header=None, mode='a')
					print("newdataset added")
				print("all data saved")

			except Exception as error:
				print(str(error))
		else:
			print("error check parameter")	
		



	def createImageDatsaset(self, dataset_name, colored_image):
		if True:
			resized_image = Image.open(colored_image)
			resized_image = resized_image.resize((self.image_size,self.image_size),Image.ANTIALIAS)
			resized_image.save("coloredImageResized.jpg",optimize=True,quality=100)

			img = Image.open("coloredImageResized.jpg").convert('LA')
			img.save('greyscale_encoded.png')
			image = imread("greyscale_encoded.png")

			print(image.ndim)

			flattent_image_array = self.toOneDimArray(image)

			array_value_count = flattent_image_array.shape[0]
			dataset_array = np.array([])
			loop_count = self.data_loop_count

			while loop_count > 0:
				dataset_array = np.append(dataset_array, flattent_image_array)
				loop_count -= 1
				print("looped" + str(loop_count))

				if loop_count == 0:
					break

			print(dataset_array.shape)
			dataset_array = dataset_array.reshape(self.data_loop_count, int(array_value_count / 2) * 2)

			print("shape" + str(dataset_array.shape))
			dataset = pd.DataFrame(dataset_array, columns = map(str, range(int(array_value_count / 2) * 2))) 
			dataset["classification"] = "normal"

			dataset.to_csv(dataset_name, index = False)
			print("newdataset added")

		else:
			print("good")



	def createDatasetByCV(self, dataset_name, add_data = False, data_classification = "normal"):
		cap = cv.VideoCapture(0)

		if not cap.isOpened():
		    print("Cannot open camera")
		    exit()

		while True:
		    ret, frame = cap.read()
		    if not ret:
		        print("Can't receive frame (stream end?). Exiting ...")
		        break

		    rescaled_frame = cv.resize(frame, (self.image_size, self.image_size))
		    normal_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		    detector_frame = cv.cvtColor(rescaled_frame, cv.COLOR_BGR2GRAY)

		    cv.imshow('Detector Frame', detector_frame)
		    cv.imshow('normal viewport', normal_frame)

		    if cv.waitKey(1) == ord('q'):
		        break

		    elif cv.waitKey(1) == ord('c'):
		    	print("please wait")
		    	cv.imwrite("cvcaptureddata.png", detector_frame)

		    	print("---" * 20)

		    	flattent_image_array = self.toOneDimArray(detector_frame)
		    	array_value_count = flattent_image_array.shape[0]
		    	dataset_array = np.array([])
		    	loop_count = self.data_loop_count

		    	while loop_count > 0:
		    		dataset_array = np.append(dataset_array, flattent_image_array)
		    		loop_count -= 1
		    		print("looped" + str(loop_count))

		    		if loop_count == 0:
		    			break

		    	print(dataset_array.shape)
		    	dataset_array = dataset_array.reshape(self.data_loop_count, int(array_value_count / 2) * 2)
		    	print("shape" + str(dataset_array.shape))
		    	dataset = pd.DataFrame(dataset_array, columns = map(str, range(int(array_value_count / 2) * 2))) 

		    	if add_data == True:
			    	dataset["classification"] = data_classification
			    	dataset.to_csv(dataset_name, index = False, header=None, mode='a')
			    	print("newdata added to dataset")

		    	elif add_data == False:
			    	dataset["classification"] = "normal"
			    	dataset.to_csv(dataset_name, index = False)
			    	print("newdataset added")
		    	break


	def AnomalyDetectByHogImage(self, image):
		try:
			resized_image = Image.open(image)
			resized_image = resized_image.resize((self.image_size,self.image_size),Image.ANTIALIAS)
			resized_image.save("deleteThisAfterDatasetIsCreated.jpg",optimize=True,quality=100)
			image = "deleteThisAfterDatasetIsCreated.jpg"

			hog_encoded_image = self.encodeHog(image)
			image_hog_data = np.array(hog_encoded_image).reshape(1, self.image_size * self.image_size)

			X = self.dataset.iloc[:, :-1].values
			y = self.dataset["classification"]

			knn = KNeighborsClassifier(n_neighbors = self.neighbors)
			knn.fit(X, y)
			prediction = knn.predict(image_hog_data)
			print(prediction)

		except Exception as error:
			print(str(error))



	def anomalyDetectByCV(self, dataset, cap_anomaly = True, absolute_changes_detection = False):
		try:
			print("loading dataset")
			print("---" * 20)
			dataset = pd.read_csv(dataset)
			print("dataset loaded complete")
			print("---" * 20)
		except Exception as error:
			print("something wrong occured while reading dataset" + str(error))

		cap = cv.VideoCapture(0)

		if not cap.isOpened():
		    print("Cannot open camera")
		    exit()

		count = 0
		while True:
		    ret, frame = cap.read()
		    if not ret:
		        print("Can't receive frame (stream end?). Exiting ...")
		        break

		    rescaled_frame = cv.resize(frame, (self.image_size, self.image_size))
		    normal_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		    detector_frame = cv.cvtColor(rescaled_frame, cv.COLOR_BGR2GRAY)

		    flattent_image_array = self.toOneDimArray(detector_frame)
		    prediction = self.knnAlgorithmDetector(dataset, [flattent_image_array])

		    if prediction == "anomaly":
		    	print("captured anomaly")

		    	if cap_anomaly == True:
			    	if not os.path.exists("anomalyImages"):
			    		os.mkdir("anomalyImages")
			    		print("folder created")

			    	count += 1
			    	cv.imwrite("anomalyImages/" + str(time.time()) + ".png" , detector_frame)

		    elif prediction == "normal":
		    	print("normal")

		    cv.imshow('Detector Frame', detector_frame)
		    cv.imshow('normal viewport', normal_frame)

		    if cv.waitKey(1) == ord('q'):
		    	flattent_image_array = self.toOneDimArray(detector_frame)
		    	print(flattent_image_array.ndim)
		    	print(flattent_image_array.shape)
		    	break
		
		cap.release()
		cv.destroyAllWindows()

