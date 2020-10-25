from tkinter import Tk, Button, Frame, Entry, OptionMenu, StringVar, Label, END
import sys
import Starguard as stg


class MainApp():
	def __init__(self):
		self.root = Tk()
		self.root.title("StarGuard")
		self.stg = stg.StarGuard()
		self.nav_area_bgcolor = "#dbdbdb"
		self.inteface_frame_bgcolor = "#f2f2f2"

		self.intro_text = """StarGuard is research project to demonstrate how machine learning
can be use for astronomy
For documentation please proceed to the project repository"""
		self.version = "0.1.6 on 10/24/2020"


	def showMainFrames(self):
		main_frame = Frame(self.root, height = "200px", width = "400px")
		main_frame.grid()

		self.nav_buttons_frame = Frame(main_frame, height = "200px", width = "100px", bg = self.nav_area_bgcolor)
		self.nav_buttons_frame.place(x = "0px", y = "0px")

		cv_option_btn = Button(self.nav_buttons_frame, text = "cv detect", width = 16, bd = "0px", activebackground = "red",command = lambda: self.cvDetection())
		cv_option_btn.place(x = "5px", y = "5px")

		single_image_detection_btn = Button(self.nav_buttons_frame, text = "image detect", width = 16, bd = "0px", activebackground = "red",command = lambda: self.singleImageAnomaly())
		single_image_detection_btn.place(x = "5px", y = "25px")

		tools_btn = Button(self.nav_buttons_frame, text = "Tools", width = 16, bd = "0px", activebackground = "red",command = lambda: self.tools())
		tools_btn.place(x = "5px", y = "45px")

		exit_btn = Button(self.nav_buttons_frame, text = "Exit", width = 16, bd = "0px", bg = "red", activebackground = "red",command = lambda: self.closeProgram())
		exit_btn.place(x = "5px", y = "180px")


		self.inteface_frame = Frame(main_frame, height = "200px", width = "300px", bg = self.inteface_frame_bgcolor)
		self.inteface_frame.place(x = "100px", y = "0px")

		Label(self.inteface_frame, text = self.intro_text, justify = "left").place(x = "5px", y ="20px")
		Label(self.inteface_frame, text = "v." + self.version).place(x = "5px", y = "170px")

		self.root.resizable(0,0)
		self.root.mainloop()


	def cvDetection(self):
		self.cv_detection_frame = Frame(self.inteface_frame, height = "200px", width = "300px", bg = self.inteface_frame_bgcolor)
		self.cv_detection_frame.place(x = "0px", y = "0px")

		Label(self.cv_detection_frame, text = "dataset:").place(x = "5", y = "5px")
		cv_dataset = Entry(self.cv_detection_frame)
		cv_dataset.place(x = "90px", y = "5px")
		cv_dataset.insert(END, "*.csv", )

		cap_anomaly = StringVar(self.cv_detection_frame)
		cap_anomaly.set("true")

		Label(self.cv_detection_frame, text = "auto capture:").place(x = "5", y = "20px")
		cap_anomaly_option_anomaly = OptionMenu(self.cv_detection_frame, cap_anomaly, "false", "true")
		cap_anomaly_option_anomaly.place(x = "90px", y = "20px")


		abs_detect = StringVar(self.cv_detection_frame)
		abs_detect.set("false")
		Label(self.cv_detection_frame, text = "abs detect:").place(x = "5", y = "40px")
		abs_detect_option_menu = OptionMenu(self.cv_detection_frame, abs_detect, "false", "true")
		abs_detect_option_menu.place(x = "90px", y = "40px")

		start_detection = Button(self.cv_detection_frame, text = "start detection", activebackground = "red",command = lambda: startDetection())
		start_detection.place(x = "220px", y = "5px")



		Label(self.cv_detection_frame, text = "Create or add data to Dataset:").place(x = "5px", y = "80px")

		Label(self.cv_detection_frame, text = "New datset name").place(x = "5px", y = "100px")
		new_dataset_name = Entry(self.cv_detection_frame)
		new_dataset_name.place(x = "90px", y = "100px")
		new_dataset_name.insert(END, "*.csv", )

		add_data_bool = StringVar(self.cv_detection_frame)
		add_data_bool.set("false")

		Label(self.cv_detection_frame, text = "existing file:").place(x = "5px", y = "120px")
		add_data_option_menu = OptionMenu(self.cv_detection_frame, add_data_bool, "true", "false")
		add_data_option_menu.place(x = "90px", y = "120px")
		
		data_class = StringVar(self.cv_detection_frame)
		data_class.set("normal")

		Label(self.cv_detection_frame, text = "classification:").place(x = "5px", y = "140px")
		data_class_option_menu = OptionMenu(self.cv_detection_frame, data_class, "normal", "anomaly")
		data_class_option_menu.place(x = "90px", y = "140px")


		crete_dataset_btn = Button(self.cv_detection_frame, text = "Create Dataset", activebackground = "red",command = lambda: createCvDataset())
		crete_dataset_btn.place(x = "220px", y = "100px")

		def startDetection():
			print("to close: Click the video viewport and press the letter x on your keyboard")
			cv_input_dataset = cv_dataset.get()
			cap_anomaly_input = cap_anomaly.get()
			if cap_anomaly_input == "true":
				cap_param_value = True
			else:
				cap_param_value = False

			abs_detect_input = abs_detect.get()
			if abs_detect_input == 'true':
				abs_detect_param_value = True
			else:
				abs_detect_param_value = False

			self.stg.anomalyDetectByCV(cv_input_dataset, cap_param_value, abs_detect_param_value)


		def createCvDataset():
			new_dataset_name_value = new_dataset_name.get()

			if len(new_dataset_name_value) <= 4:
				print("datset name is too short")
			else:
				add_data_bool_value = add_data_bool.get()
				if add_data_bool_value == "true":
					add_data = True
				elif add_data_bool_value == "false":
					add_data = False
				data_class_value = data_class.get()

				self.stg.createDatasetByCV(new_dataset_name_value, add_data, data_class_value)


	def singleImageAnomaly(self):
		self.single_image_frame = Frame(self.inteface_frame, height = "200px", width = "300px", bg = self.inteface_frame_bgcolor)
		self.single_image_frame.place(x = "0px", y = "0px")

		Label(self.single_image_frame, text = "sample image:").place(x = "0px", y = "5px")
		test_image = Entry(self.single_image_frame)
		test_image.place(x = "70px", y = "5px")

		Label(self.single_image_frame, text = "datset:").place(x = "0px", y = "25pxpx")
		dataset = Entry(self.single_image_frame)
		dataset.place(x = "70px", y = "25px")
		dataset.insert(END, "*.csv")

		predict_button = Button(self.single_image_frame, text = "start prediction", activebackground = "red", command = lambda: startPrediction())
		predict_button.place(x = "220px", y = "5px")


		Label(self.single_image_frame, text = "Create Dataset").place(x = "0px", y = "60px")
		Label(self.single_image_frame, text = "image:").place(x = "0px", y = "80px")
		new_data_image = Entry(self.single_image_frame)
		new_data_image.place(x = "70px", y = "80px")

		Label(self.single_image_frame, text = "datset name:").place(x = "0px", y = "100px")
		new_dataset = Entry(self.single_image_frame)
		new_dataset.place(x = "70px", y = "100px")
		new_dataset.insert(END, "*.csv")

		image_class = StringVar(self.single_image_frame)
		image_class.set("normal")

		Label(self.single_image_frame, text = "Image class:").place(x = "5px", y = "120px")
		image_class_option_menu = OptionMenu(self.single_image_frame, image_class, "normal", "anomaly")
		image_class_option_menu.place(x = "70px", y = "120px")

		dataset_bool = StringVar(self.single_image_frame)
		dataset_bool.set("true")

		Label(self.single_image_frame, text = "new dataset:").place(x = "5px", y = "145px")
		dataset_bool_option_menu = OptionMenu(self.single_image_frame, dataset_bool, "true", "false")
		dataset_bool_option_menu.place(x = "70px", y = "145px")

		edit_dataset_button = Button(self.single_image_frame, text = "process datset", activebackground = "red", command = lambda: processDataset())
		edit_dataset_button.place(x = "220px", y = "80px")

		def processDataset():
			dataset_bool_input = dataset_bool.get()
			new_dataset_input =  new_dataset.get()
			new_data_image_input = new_data_image.get()
			image_class_input = image_class.get()

			if image_class_input == "normal":
				normal = True
			elif image_class_input == "anomaly":
				normal = False

			if dataset_bool_input == "true":
				self.stg.createImageHogDataset(new_dataset_input, new_data_image_input)

			elif dataset_bool_input == "false":
				self.stg.addDataToImageHogDataset(new_dataset_input, new_data_image_input, normal)


		def startPrediction():
			test_image_input = test_image.get()
			dataset_input = dataset.get()

			self.stg.AnomalyDetectByHogImage(test_image_input, dataset_input)



	def tools(self):
		self.tools_frame = Frame(self.inteface_frame, height = "200px", width = "300px", bg = self.inteface_frame_bgcolor)
		self.tools_frame.place(x = "0px", y = "0px")

		Label(self.tools_frame, text = "dataset").place(x = "5px", y = "5px")
		dataset_entry = Entry(self.tools_frame)
		dataset_entry.place(x = "70px", y = "5px")


		view_dataset_accuracy_btn = Button(self.tools_frame, text = "check accuracy", command = lambda: self.stg.testDatasetAccuracy(dataset_entry.get()))
		view_dataset_accuracy_btn.place(x = "5px", y = "25px")

		view_dataset_btn = Button(self.tools_frame, text = "Read dataset", command = lambda: self.stg.showDataset(dataset_entry.get()))
		view_dataset_btn.place(x = "80px", y = "25px")



	def closeProgram(self):
		self.root.destroy()
		print("closing program")
		sys.exit()


			

if __name__ == "__main__":
	app = MainApp()
	app.showMainFrames()