try:
	import matplotlib.image as mpimg
	import numpy as np
	import pandas as pd
	from PIL import Image
	from skimage.feature import hog
	from skimage.io import imread
	from skimage import data, exposure
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



	def br(self):
		print("")
		print(("---" * 10) + ("END") + ("---" * 10))
		print("")
		print("")
		print("")


	def knnAlgorithmDetector(self, dataset, frame):
		X = dataset.iloc[:, :-1].values
		y = dataset["classification"]
		
		knn = KNeighborsClassifier(n_neighbors = self.neighbors)
		knn.fit(X, y)
		prediction = knn.predict(frame)
		return prediction



	def help(self):
		print("""list of methods:
			knnAlgorithmDetector(dataset name (.csv), image dir) >>> Return a prediction
			testDatasetAccuracy(dataset name (.csv)) >>> print the accuracy of the dataset
			showDataset(dataset name (.csv)) >>> print the dataset
			encodeHog(image dir) >>> return a hod encoded image
			toOneDimArray(multidimensional numpy array) >>> return a flattened version of array
			createImageHogDataset(dataset name (.csv), image dir) >>> create a dataset file
			addDataToImageHogDataset(dataset name (.csv), image dir, bool) >>> add data to the existing dataset
			createDatasetByCV(dataset name (.csv), add_data = bool, data_classification = "normal" or "anomaly str) >>> create or add new data to a dataset
			anomalyDetectByHogImage(image dir, dataset name (.csv)) >>> print a prediction. to exit this method just click the camera viewport and click the x button on keyboard
			anomalyDetectByCV(dataset name (.csv), cap_anomaly = bool, print_frame_class = bool, absolute_changes_detection = bool) >>> print a prediction. this method will open the cameraand will capture the image if parameter is true. to exit this method just click the camera viewport and click the x button on keyboard
			""")
		self.br()




	def testDatasetAccuracy(self, dataset):
		print("loading dataset")
		dataset = pd.read_csv(dataset)

		print("load complete", "calculating....")
		X = dataset.iloc[:, :-1].values
		y = dataset["classification"]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		knn = KNeighborsClassifier(n_neighbors = self.neighbors)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)

		print(y_pred)
		print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
		self.br()



	def showDataset(self, dataset):
		print("loading dataset")
		loaded_dataset = pd.read_csv(dataset)
		print(loaded_dataset)
		self.br()



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

			self.br()
		
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

			self.br()

		except Exception as error:
			print(str(error))
	


	def createDatasetByCV(self, dataset_name, add_data = False, data_classification = "normal"):
		cap = cv.VideoCapture(0)

		if not cap.isOpened():
		    print("Cannot open camera")
		    exit()

		while True:
			ret, frame = cap.read()
			if not ret:
				print("Can't any receive frame (stream end?). Exiting ...")
				break

			rescaled_frame = cv.resize(frame, (self.image_size, self.image_size))
			normal_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			detector_frame = cv.cvtColor(rescaled_frame, cv.COLOR_BGR2GRAY)

			cv.imshow('Detector Frame', detector_frame)
			cv.imshow('normal viewport', normal_frame)

			if cv.waitKey(1) == ord('x'):
				break

			elif cv.waitKey(1) == ord('c'):
				print("please wait")
				cv.imwrite("cvcaptureddata.jpg", detector_frame)

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
					self.br()

				elif add_data == False:
					dataset["classification"] = "normal"
					dataset.to_csv(dataset_name, index = False)
					print("newdataset added")
					self.br()
				break



	def anomalyDetectByHogImage(self, image, dataset):
		try:
			print("loacding dataset")
			dataset = pd.read_csv(dataset)

			resized_image = Image.open(image)
			resized_image = resized_image.resize((self.image_size,self.image_size),Image.ANTIALIAS)
			resized_image.save("deleteThisAfterDatasetIsCreated.jpg",optimize=True,quality=100)
			image = "deleteThisAfterDatasetIsCreated.jpg"

			hog_encoded_image = self.encodeHog(image)
			image_hog_data = np.array(hog_encoded_image).reshape(1, self.image_size * self.image_size)

			X = dataset.iloc[:, :-1].values
			y = dataset["classification"]

			knn = KNeighborsClassifier(n_neighbors = self.neighbors)
			knn.fit(X, y)
			prediction = knn.predict(image_hog_data)
			print(prediction)

			self.br()

		except Exception as error:
			print(str(error))



	def anomalyDetectByCV(self, dataset, cap_anomaly = True, print_frame_class = True, absolute_changes_detection = False):
		try:
			print("loading dataset")
			print("---" * 20)
			dataset = pd.read_csv(dataset)
			output = dataset.loc[dataset['classification'] == "normal"]
			self.normal_absolute_data = np.array(output.head(1))[0][0:-1]
			print("dataset loaded complete")
			print("---" * 20)

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

			    if absolute_changes_detection == False:
			    	prediction = self.knnAlgorithmDetector(dataset, [flattent_image_array])
			    elif absolute_changes_detection == True:
			    	comparison = self.normal_absolute_data == flattent_image_array
			    	if False in comparison:
			    		prediction = "anomaly"
			    	else:
			    		prediction = "normal"

			    else:
			    	print("parameter error")
			    	break

			    if prediction == "anomaly":
			    	if print_frame_class == True:
			    		print("captured anomaly")

			    	if cap_anomaly == True:
				    	if not os.path.exists("anomalyImages"):
				    		os.mkdir("anomalyImages")
				    		print("folder created")

				    	count += 1
				    	cv.imwrite("anomalyImages/" + str(time.time()) + ".jpg" , detector_frame)

			    elif prediction == "normal":
			    	if print_frame_class == True:
			    		print("normal")

			    cv.imshow('Detector Frame', detector_frame)
			    cv.imshow('normal viewport', normal_frame)

			    if cv.waitKey(1) == ord('x'):
			    	break
			
			cap.release()
			cv.destroyAllWindows()

			self.br()


		except Exception as error:
			print("Error:" + str(error))
			self.br()

if __name__ == "__main__":
	module = StarGuard()