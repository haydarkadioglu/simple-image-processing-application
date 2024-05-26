from Ui_MainWindow import Ui_MainWindow


from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt


from Functions import SIP as sip
import numpy as np
import matplotlib.pyplot as plt

from cv2 import imwrite

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # variable
        self.imageOrgPixmap = None
        self.imageVPixmap = None

        self.imageOrgArray = None
        self.imageGrayArray = None
        self.imageVArray = None
        self.secondImage = None

        self.imageVersions = []
        self.imageIndex = len(self.imageVersions) - 1

        # click events
        self.pushButton_select.clicked.connect(self.select_image)
        self.pushButton_gray.clicked.connect(self.convertGray)
        self.pushButton_binary.clicked.connect(self.convertBinary)
        self.pushButton_thresh.clicked.connect(self.single_thresh)
        self.pushButton_thresh_2.clicked.connect(self.double_thresh)
        self.pushButton_transformation.clicked.connect(self.transformation)
        self.pushButton_contrast.clicked.connect(self.increaseContrast)
        self.pushButton_rotate.clicked.connect(self.rotation)
        self.pushButton_crop.clicked.connect(self.crop)
        self.pushButton_zoom.clicked.connect(self.zoom)
        self.pushButton_mean.clicked.connect(self.applyMean)
        self.pushButton_median.clicked.connect(self.applyMedian)
        self.pushButton_saltpepper.clicked.connect(self.applySaltpepper)
        self.pushButton_unsharp.clicked.connect(self.applyUnsharp)
        self.pushButton_prewitt.clicked.connect(self.applyPrewitt)
        self.pushButton_add.clicked.connect(self.addP)
        self.pushButton_divide.clicked.connect(self.divideP)
        self.pushButton_Merode.clicked.connect(self.eroding)
        self.pushButton_Mdilate.clicked.connect(self.dilating)
        self.pushButton_Mopen.clicked.connect(self.opening)
        self.pushButton_Mclose.clicked.connect(self.closing)
        self.pushButton_histGray.clicked.connect(self.showHistGray)
        self.pushButton_histR.clicked.connect(self.showHistRed)
        self.pushButton_histG.clicked.connect(self.showHistGreen)
        self.pushButton_histB.clicked.connect(self.showHistBlue)

        self.pushButton_histEq.clicked.connect(self.equalize)
        self.pushButton_histStrech.clicked.connect(self.streching)

        self.actionUndo.triggered.connect(self.unDo)
        self.actionRedo.triggered.connect(self.reDo)
        self.actionSave.triggered.connect(self.saveImage)
        self.actionClear.triggered.connect(self.clearLabel)
        self.actionSelect_new_image.triggered.connect(self.new_image)

        self.redoShortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redoShortcut.activated.connect(self.reDo)
        self.undoShortcut = QShortcut(QKeySequence.Undo, self)
        self.undoShortcut.activated.connect(self.unDo)
        self.saveShortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.saveShortcut.activated.connect(self.saveImage)

    def new_image(self):
        self.clearLabel()
        self.select_image()

    def select_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)
            self.imageOrgPixmap = pixmap
            self.imageOrgArray = self.convert(pixmap=pixmap)

            self.imageVersions.append(self.imageOrgArray)
            self.update()

            self.pushButton_select.hide()

            self.displayImage(pixmap)

    def convert(self, pixmap=None, arrays=None):
        
        format = None
        channels = 1
            

        if pixmap is not None:
            image = pixmap.toImage()

            # QImage'den veriyi alarak bir byte dizisine dönüştürme
            byte_array = image.bits().asstring(image.byteCount())



            width, height = image.width(), image.height()
            channels = 4  
            image_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, channels))
            image_array = image_array[:,:,:3]

            return np.array(image_array)
                
        else:
            if len(arrays.shape) == 3:
                format = QImage.Format_RGB888
                channels = 3
            else:
                format = QImage.Format_Grayscale8

            height, width = arrays.shape[:2]
            bytes_per_line = channels * width
            qimage = QImage(arrays.data, width, height, bytes_per_line, format).rgbSwapped()



            pixmap = QPixmap.fromImage(qimage)

            return QPixmap.fromImage(qimage)
        


    def displayImage(self, pixmap):
        

        if not pixmap.isNull():
            pixmap_size = pixmap.size()

            groupbox_size = self.groupBox.size()

            if pixmap_size.width() <= groupbox_size.width() and pixmap_size.height() <= groupbox_size.height():
                pass
            else:
                pixmap = pixmap.scaled(groupbox_size, Qt.KeepAspectRatio)
            # pixmap = pixmap.scaled(self.groupBox.size(), Qt.KeepAspectRatio)
            self.label_image.setPixmap(pixmap)
            self.label_image.setFixedSize(pixmap.size())
                
            self.label_image.setAlignment(Qt.AlignCenter)


        else:
            QMessageBox.information(self, "Error", "Invalid file")
    

    

    def saveImage(self):

        filename, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if filename:
            if len(self.imageVArray)==3:
                plt.imsave(filename, self.imageVArray)
            else: 
                rgb_image = np.stack((self.imageVArray,) * 3, axis=-1)
                plt.imsave(filename, rgb_image)


    def clearLabel(self):
        self.label_image.clear()
        self.pushButton_select.show()
        self.imageVersions.clear()
        self.label_image.setFixedSize(self.groupBox.size())



    def exit(self):
        pass

    def unDo(self):
        if self.imageIndex > 0:
            self.imageIndex -= 1
            self.imageVArray = self.imageVersions[self.imageIndex]
            self.displayImage(pixmap=self.convert(arrays=self.imageVArray))

        else:
            self.messageBox("There is no image to undo.")

    def reDo(self):
        if self.imageIndex < (len(self.imageVersions)-1):
            self.imageIndex += 1
            self.imageVArray = self.imageVersions[self.imageIndex]
            self.displayImage(pixmap=self.convert(arrays=self.imageVArray))

        else:
            self.messageBox("There is no image to redo.")

    def update(self):
        self.imageIndex = len(self.imageVersions) - 1
        self.imageVArray = self.imageVersions[self.imageIndex]

    def convertGray(self):

        img = self.imageVArray
        che = self.check()
        if che:return
        img = sip.Colors.convert_to_gray(img)



        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageGrayArray = img
        self.imageVersions.append(img)
        self.update()


    def convertBinary(self):
        img = self.imageGrayArray
        che = self.check(isgray=True, isempyt=True, isint=True,input=self.lineEdit_binary.text())
        if che:return
        img = sip.Colors.convert_to_binary(img, int(self.lineEdit_binary.text()))

       
        self.displayImage(pixmap=self.convert(arrays=img))
        
        self.imageVersions.append(img)
        self.update()

    def single_thresh(self):
        img = self.imageGrayArray
        che = self.check(isgray=True, isempyt=True, isint=True,input=self.lineEdit_binary.text())
        if che:return
        img = sip.Colors.single_threshold(img, int(self.lineEdit_thresh.text()))

       
        self.displayImage(pixmap=self.convert(arrays=img))
        
        self.imageVersions.append(img)
        self.update()

    def double_thresh(self):
        img = self.imageGrayArray
        che = self.check(isgray=True, isempyt=True, islist=True,lenght=3,input=self.lineEdit_thresh_2.text().split(","))
        if che:return
        values = self.lineEdit_thresh_2.text().split(",")
        img = sip.Colors.double_threshold(img, int(values[0]), int(values[1]), int(values[2]))

       
        self.displayImage(pixmap=self.convert(arrays=img))
        
        self.imageVersions.append(img)
        self.update()

    def transformation(self):
        img = self.imageOrgArray
        che = self.check(isempyt=True, islist=True, input=self.lineEdit_transformation.text().split(","), lenght=3)
        if che:return
        values = self.lineEdit_transformation.text().split(",")
        img = sip.Colors.rgb_transformation(img, b=int(values[2]), g=int(values[1]), r=int(values[0]))

        self.displayImage(pixmap=self.convert(arrays=img))
        self.imageVersions.append(img)
        self.update()


    def increaseContrast(self):
    
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_contrast.text(), isfloat=True)
        if che:return
        value = float(self.lineEdit_contrast.text())

        img = sip.Colors.increase_contrast(img, value)
        
        self.displayImage(pixmap=self.convert(arrays=img))
        self.imageVersions.append(img)
        self.update()
    
        

    def rotation(self):
        img = self.imageVArray
        che = self.check()
        if che:return
        img = sip.Rotate.rotate_image(img)
        self.displayImage(pixmap=self.convert(arrays=img))
        self.imageVersions.append(img)
        self.update()



    def crop(self):
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_crop.text(), islist=True, lenght=4)
        values = self.lineEdit_crop.text().split(",")
        che2 = self.check(input=values[0], isint=True)
        che3 = self.check(input=values[1], isint=True)
        che4 = self.check(input=values[2], isint=True)
        che5 = self.check(input=values[3], isint=True)
        if che and che2 and che3 and che4 and che5:return
        values = self.lineEdit_crop.text().split(",")
        

        img = sip.Rotate.crop(img, int(values[0]), int(values[1]), int(values[2]), int(values[3]))
        self.displayImage(pixmap=self.convert(arrays=img))
        self.imageVersions.append(img)
        self.update()

    def zoom(self):
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_zoom.text(), isfloat=True)
        if che:return
        value = self.lineEdit_zoom.text()
        img = sip.Rotate.zoom(img, float(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()


    # Tab filters
    def applyMean(self):
        
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_mean.text(), isint=True)
        if che:return
        value = self.lineEdit_mean.text()


        img = sip.Filters.mean_filter(img,int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()



    def applyMedian(self):
        
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_median.text(), isint=True)
        if che:return
        value = self.lineEdit_median.text()


        img = sip.Filters.median_filter(img,int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()
        
    def applySaltpepper(self):
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_saltpepper.text(), isint=True)
        if che:return
        value = self.lineEdit_saltpepper.text()


        img = sip.Filters.salt_pepper(img,int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()
    
    def applyUnsharp(self):
        img = self.imageVArray
        che = self.check(isempyt=True, input=self.lineEdit_unsharp.text(), islist=True)
        if che:return
        values = self.lineEdit_unsharp.text().split(",")
        che1 = self.check(input=values[0], isint=True)
        che2 = self.check(input=values[1], isfloat=True)
        if che1 and che2: return


        img = sip.Filters.unsharp_mask(img,float(values[0]),float(values[1]))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def applyPrewitt(self):
        img = self.imageVArray
        che = self.check(isgray=True)
        if che:return


        img = sip.Filters.detect_edge_prewitt(img)

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()       

    def addP(self):
        
        
        values = self.lineEdit_add.text().split(",")
        che = self.check(input=self.lineEdit_add.text().split(","), islist=True, lenght=2)
        che1 = self.check(input=values[0], isfloat=True)
        che2 = self.check(input=values[1], isfloat=True)
        if che:return
        filename, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)            
            self.secondImage = self.convert(pixmap=pixmap)
        img = self.imageVArray
        img = sip.Aritmatich.add_weighted(img, float(values[0]), self.secondImage, float(values[0]))
        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def divideP(self):
        che = self.check()
        if che:return
        filename, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)            
            self.secondImage = self.convert(pixmap=pixmap)

        
        img = self.imageVArray
        img = sip.Aritmatich.divide(img, self.secondImage)
        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    # Tab Morph&Hist

    def eroding(self):
        img = self.imageVArray
        che = self.check(isempyt=True,isgray=True, input=self.lineEdit_Merode.text(), isint=True)
        if che:return
        value  = self.lineEdit_Merode.text()

        img = sip.Histogram.erode(self.imageVArray, int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def dilating(self):
        img = self.imageVArray
        che = self.check(isempyt=True,isgray=True, input=self.lineEdit_Mdilate.text(), isint=True)
        if che:return
        value  = self.lineEdit_Mdilate.text()

        img = sip.Histogram.dilate(self.imageVArray, int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def opening(self):
        img = self.imageVArray
        che = self.check(isempyt=True,isgray=True, input=self.lineEdit_Mopen.text(), isint=True)
        if che:return
        value  = self.lineEdit_Mopen.text()

        img = sip.Histogram.opening(self.imageVArray, int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def closing(self):
        img = self.imageVArray
        che = self.check(isempyt=True,isgray=True, input=self.lineEdit_Mclose.text(), isint=True)
        if che:return
        value  = self.lineEdit_Mclose.text()

        img = sip.Histogram.closing(self.imageVArray, int(value))

        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def equalize(self):
        img = self.imageVArray
        che = self.check(isgray=True)
        if che:return
        img = sip.Histogram.histogram_equalization(img)
        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def streching(self):
        img = self.imageVArray
        che = self.check(isgray=True)
        if che:return
        img = sip.Histogram.histogram_stretching(img)
        self.displayImage(pixmap=self.convert(arrays=img))

        self.imageVersions.append(img)
        self.update()

    def showHistGray(self):
        img = self.imageVArray
        che = self.check(isgray=True)
        if che:return
        value = sip.Histogram.calculate_gray_histogram(img)
        self.plot_histogram(value, "black")
    def showHistRed(self):
        img = self.imageVArray
        che = self.check()
        if che:return
        value = sip.Histogram.calculate_rgb_histogram(img)[0]
        self.plot_histogram(value,"red")
    def showHistGreen(self):
        img = self.imageVArray
        che = self.check()
        if che:return
        value = sip.Histogram.calculate_rgb_histogram(img)[1]
        self.plot_histogram(value, "green")
    def showHistBlue(self):
        img = self.imageVArray
        che = self.check()
        if che:return
        value = sip.Histogram.calculate_rgb_histogram(img)[2]
        self.plot_histogram(value, "blue")

    def plot_histogram(self, histogram, color):
        plt.figure()
        plt.bar(range(256), histogram, width=1.0, color=color)
        plt.title("Gray Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.show()


    def check(self, isgray=False, isempyt=False, input = None, 
              islist=False,
              isint = False,
              isfloat=False,
              lenght:int=0
              ):
        if len(self.imageVersions)==0 and self.imageOrgArray == None:return self.messageBox("You have to select an image...")
        if isgray and self.imageGrayArray is None:return self.messageBox("You have to convert image to gray scale...")
        if isempyt and input == "":return self.messageBox("You have to input a value...")
        #if islist and type(input)!=list():return self.messageBox("You have to input a list...")
        if islist and lenght!=0 and len(input) != lenght:return self.messageBox(f"You have to write {lenght} integer parameters like how it is shown...")
        if isfloat:
            try: float(input) 
            except: return self.messageBox("You have to write float type value...")
        if isint:
            try: int(input) 
            except: return self.messageBox("You have to write integer type value...")


        return False
    def messageBox(self, message):
        QMessageBox.warning(self,"WARNING", message)
        return True

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
