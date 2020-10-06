try:
	import matplotlib.image as mpimg
	import numpy as np
	import pandas as pd
	from PIL import Image
	from skimage.feature import hog
	from skimage.io import imread
	from skimage import data, exposure
	from astropy.io import fits
	
except Exception as error:
	print("Library import error: " + str(error))
	print("if you think this is a bug please open a new issue in project repository")



class starguard():
	def __init__(self, dataset, image_size = 200):
		self.dataset = dataset
		self.image_size = image_size
		self.data_loop_count = 20



	def encodeHog(self, image_input):
	    image = imread(image_input)
	    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(2, 2), 
	    	visualize = True, multichannel = True)

	    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
	    return hog_image_rescaled
		    


	def toOneDimArray(self, multi_dimensional_array):
		flattend_array = multi_dimensional_array.flatten()
		return flattend_array



	def createImageDataset(self, dataset_name, image):
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
			dataset["name"] = "normal"

			dataset.to_csv(dataset_name, index = False)
			print("newdataset added")

			
		except Exception as error:
			print("Error:" +  str(error))



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
			dataset["name"] = "normal"

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
						dataset["name"] = "normal"
					elif normal == False:
						dataset["name"] = "anomaly"
					else:
						print("runtime error")

					dataset.to_csv(target_dataset, index = False, header=None, mode='a')
					print("newdataset added")

				print("all data saved")
					

			except Exception as error:
				print(str(error))

		else:
			print("error check parameter")	
		


	def createColorBaseDatsaset(self):
		pass


	def showDataset(self):
		dataset = pd.read_csv(self.dataset)
		print(dataset)


	def anomalyDetectByFits(self):
		pass



	def AnomalyDetectByBWImage(self):
		pass



	def AnomalyDetectByCV(self):
		pass



	def AnomalyDetectByVideo(self):
		pass



	def PerfomanceEval(self):
		pass


Alice = starguard("fitsdataset.csv")
#Alice.createImageDataset("sampleset.csv", "sample.jpg")
#Alice.createDatasetFromFits("fitsdataset.csv", "sampfits.fits")

Alice.addDataToFitsDataSet("fitsdataset.csv", ["sampfits.fits"] ,True)
Alice.showDataset()