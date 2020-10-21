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
		cv_dataset.place(x = "70px", y = "5px")

		cap_anomaly = StringVar(self.cv_detection_frame)
		cap_anomaly.set("true")

		Label(self.cv_detection_frame, text = "auto capture:").place(x = "5", y = "20px")
		cap_anomaly_option_anomaly = OptionMenu(self.cv_detection_frame, cap_anomaly, "false", "true")
		cap_anomaly_option_anomaly.place(x = "70px", y = "20px")


		abs_detect = StringVar(self.cv_detection_frame)
		abs_detect.set("true")
		Label(self.cv_detection_frame, text = "abs detect:").place(x = "5", y = "40px")
		abs_detect_option_menu = OptionMenu(self.cv_detection_frame, abs_detect, "false", "true")
		abs_detect_option_menu.place(x = "70px", y = "40px")

		start_detection = Button(self.cv_detection_frame, text = "start detection", command = lambda: startDetection())
		start_detection.place(x = "5px", y = "160px")

		def startDetection():
			cv_input_dataset = cv_dataset.get()
			self.stg.anomalyDetectByCV(cv_input_dataset)





	def closeProgram(self):
		self.root.destroy()
		print("closing program")
		sys.exit()


			


app = MainApp()
app.showMainFrames()
	