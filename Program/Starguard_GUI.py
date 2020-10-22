from tkinter import Tk, Button, Frame, Entry, OptionMenu, StringVar, Label
import sys
import Starguard as stg


class MainApp():
	def __init__(self):
		self.root = Tk()
		self.stg = stg.StarGuard()
		self.nav_area_bgcolor = "#dbdbdb"
		self.inteface_frame_bgcolor = "#f2f2f2"


	def showMainFrames(self):
		main_frame = Frame(self.root, height = "200px", width = "400px")
		main_frame.grid()

		self.nav_buttons_frame = Frame(main_frame, height = "200px", width = "100px", bg = self.nav_area_bgcolor)
		self.nav_buttons_frame.place(x = "0px", y = "0px")

		cv_option_btn = Button(self.nav_buttons_frame, text = "cv detect", width = 16, bd = "0px", command = lambda: self.cvDetection(	))
		cv_option_btn.place(x = "5px", y = "5px")

		exit_btn = Button(self.nav_buttons_frame, text = "Exit", width = 16, bd = "0px", bg = "red", command = lambda: self.closeProgram())
		exit_btn.place(x = "5px", y = "180px")


		self.inteface_frame = Frame(main_frame, height = "200px", width = "300px", bg = self.inteface_frame_bgcolor)
		self.inteface_frame.place(x = "100px", y = "0px")

		self.root.mainloop()


	def cvDetection(self):
		self.cv_detection_frame = Frame(self.inteface_frame, height = "200px", width = "300px", bg = self.inteface_frame_bgcolor)
		self.cv_detection_frame.place(x = "0px", y = "0px")

		Label(self.cv_detection_frame, text = "dataset:").place(x = "5", y = "5px")
		cv_dataset = Entry(self.cv_detection_frame)
		cv_dataset.place(x = "90px", y = "5px")

		cap_anomaly = StringVar(self.cv_detection_frame)
		cap_anomaly.set("true")

		Label(self.cv_detection_frame, text = "auto capture:").place(x = "5", y = "20px")
		cap_anomaly_option_anomaly = OptionMenu(self.cv_detection_frame, cap_anomaly, "false", "true")
		cap_anomaly_option_anomaly.place(x = "90px", y = "20px")


		abs_detect = StringVar(self.cv_detection_frame)
		abs_detect.set("true")
		Label(self.cv_detection_frame, text = "abs detect:").place(x = "5", y = "40px")
		abs_detect_option_menu = OptionMenu(self.cv_detection_frame, abs_detect, "false", "true")
		abs_detect_option_menu.place(x = "90px", y = "40px")

		start_detection = Button(self.cv_detection_frame, text = "start detection", command = lambda: startDetection())
		start_detection.place(x = "220px", y = "5px")



		Label(self.cv_detection_frame, text = "Create or add data to Dataset:").place(x = "5px", y = "80px")

		Label(self.cv_detection_frame, text = "New datset name").place(x = "5px", y = "100px")
		new_dataset_name = Entry(self.cv_detection_frame)
		new_dataset_name.place(x = "90px", y = "100px")

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


		crete_dataset_btn = Button(self.cv_detection_frame, text = "Create Dataset", command = lambda: createCvDataset())
		crete_dataset_btn.place(x = "220px", y = "100px")

		def startDetection():
			print("to close: Click the video viewport and press the letter q on your keyboard")
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
			add_data_bool_value = add_data_bool.get()
			if add_data_bool_value == "true":
				add_data = True
			elif add_data_bool_value == "false":
				add_data = False
			data_class_value = data_class.get()

			self.stg.createDatasetByCV(new_dataset_name_value, add_data, data_class_value)





	def closeProgram(self):
		self.root.destroy()
		print("closing program")
		sys.exit()


			


app = MainApp()
app.showMainFrames()
	