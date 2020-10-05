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

			empty_dataset_array = np.array([])
			dataset_array = np.append(empty_dataset_array, image_hog_data)
			dataset_array = dataset_array.reshape(1, self.image_size * self.image_size)

			print("shape" + str(dataset_array.shape))
			dataset = pd.DataFrame(dataset_array, columns = map(str, range(self.image_size * self.image_size))) 
			dataset["name"] = "positive"

			dataset.to_csv(dataset_name, index = False)
			print("newdataset added")


		except Exception as error:
			print("Error:" +  str(error))


	def createDatasetFromFits(self, dataset_name,fits_image):
		fits_image = fits_image
		fits_data, self.header = fits.getdata(fits_image, ext=0, header = True)

		fits_array = np.array(fits_data[0]).reshape(1, 1600)
		array_shape = fits_array.shape
		fits_array.reshape(1, array_shape[1])
		
		print(array_shape)

		dataset = pd.DataFrame(fits_array, columns =map(str, range(int(array_shape[1] / 2) * 2))) 
		dataset["name"] = "positive"

		dataset.to_csv(dataset_name, index = False)
		print("newdataset added")



	def createColorBaseDatsaset(self):
		pass


	def showDataset(self):
		dataset = pd.read_csv(self.dataset)
		print(dataset)


	def AnomalyDetectByBWImage(self):
		pass



	def AnomalyDetectByCV(self):
		pass



	def AnomalyDetectByVideo(self):
		pass



	def PerfomanceEval(self):
		pass


Alice = starguard("sampleset.csv")
Alice.showDataset()