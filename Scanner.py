import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.draw as draw
import skimage.exposure as exposure

import pydicom
import datetime
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset


class Scanner:

    def __init__(self, nDet=None, phi=None, r=None, noViews=None):
        self.__no_detectors = nDet
        self.__phi = phi
        self.__radius = r
        self.__no_views = noViews
        self.__img = np.array([])
        self.__spect = np.array([])
        self.__restored_img = np.array([])
        self.__filter = np.array([])

    def load_image(self, img: np.ndarray):
        self.__img = img
        self.__restored_img = np.zeros(img.shape, dtype=float)
        self.__radius = max(img.shape[0], img.shape[1]) / 2

    def set_no_detectors(self, nDet: int):
        self.__no_detectors = nDet

    def set_phi(self, phi: float):
        self.__phi = phi

    def set_radius(self, radius: float):
        self.__radius = radius

    def set_no_views(self, no_views: int):
        self.__no_views = no_views

    def gen_filter(self, n: int):
        fil = [1]
        for i in range(1, n//2):
            if i % 2 == 1:
                v = -4 / np.pi**2 / i**2
                fil = [v] + fil + [v]
            else:
                fil = [0] + fil + [0]

        self.__filter = np.array(fil)

    def get_spect(self):
        return self.__spect

    def get_restored_img(self):
        return self.__restored_img

    def __emit_det(self, alpha: float) -> tuple:
        det = []

        size = self.__img.shape

        middle = (size[0] / 2, size[1] / 2)

        emit = (middle[0] + self.__radius * np.cos(alpha), middle[1] - self.__radius * np.sin(alpha))

        for i in range(self.__no_detectors):
            tmp = (middle[0] + self.__radius * np.cos(alpha + np.pi - self.__phi / 2 +
                                                      i * self.__phi / (self.__no_detectors - 1)),
                   middle[1] - self.__radius * np.sin(alpha + np.pi - self.__phi / 2 +
                                                      i * self.__phi / (self.__no_detectors - 1)))
            det.append(tmp)

        return emit, det

    def __reconstruct_view(self, lines: list, sin_vals: np.ndarray):
        for i in range(len(lines)):
            self.__reconstruct_line(lines[i], sin_vals[i])

    def __reconstruct_line(self, line: tuple, sin_val: float):
        for i in range(len(line[0])):
            if 0 < line[0][i] < self.__img.shape[0] and 0 < line[1][i] < self.__img.shape[1]:
                self.__restored_img[line[0][i], line[1][i]] += sin_val

    def __norm_img(self):
        flat = self.__restored_img.flatten()
        perc = np.percentile(flat, [5, 98])
        self.__restored_img = exposure.rescale_intensity(self.__restored_img, in_range=(perc[0], perc[1]),
                                                         out_range=(0.0, 1.0))

    def __radon_line(self, line): # to be used in a thread to speed things up
        cumBrightness = 0.0
        noElements = 0

        for k in range(len(line[0])):
            if 0 < line[0][k] < self.__img.shape[0] and 0 < line[1][k] < self.__img.shape[1]:
                cumBrightness += self.__img[line[0][k], line[1][k]]
                noElements += 1

        if noElements > 0:
            return cumBrightness / noElements
        else:
            return 0.0

    def genSpect(self, no_views: int):
        self.__spect = []
        spect = []
        self.__restored_img = np.zeros(self.__restored_img.shape)

        for i in range(no_views):
            ed = self.__emit_det(i * 2 * np.pi / self.__no_views)

            emit = ed[0]
            det = ed[1]

            view = []
            lines = []
            for j in det:
                line = draw.line_nd(emit, j)
                lines.append(line)

                cumBrightness = 0.0
                noElements = 0

                for k in range(len(line[0])):
                    if 0 < line[0][k] < self.__img.shape[0] and 0 < line[1][k] < self.__img.shape[1]:
                        cumBrightness += self.__img[line[0][k], line[1][k]]
                        noElements += 1

                if noElements > 0:
                    view.append(cumBrightness / noElements)
                else:
                    view.append(0.0)

            view = np.convolve(view, self.__filter, mode='same')
            self.__reconstruct_view(lines, view)
            spect.append(view)

        self.__spect = np.array(spect)
        self.__norm_img()

    def save_to_dicom(self, filename: str, patient_name: str, patient_sex: str, patient_age: str,
                      patient_id: str, scan_date: datetime.datetime, image_comments: str):
        if filename[-4:] != ".dcm":
            filename += ".dcm"

        # meta data for DICOM file
        meta = FileMetaDataset()

        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"

        ds = FileDataset(filename, {}, file_meta=meta, preamble=b'\0' * 128)

        # basic patient data
        ds.PatientName = patient_name
        ds.PatientAge = patient_age
        ds.PatientSex = patient_sex
        ds.PatientID = patient_id

        # more meta - required to save properly
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # date and time of diagnosis
        ds.ContentDate = scan_date.strftime('%Y%m%d')
        ds.ContentTime = scan_date.strftime('%H%M%S')

        ds.ImageComments = image_comments

        # save image to DICOM
        ds.PixelData = self.__restored_img.tobytes()
        ds.Rows = self.__restored_img.shape[1]
        ds.Columns = self.__restored_img.shape[0]
        ds.BitsAllocated = self.__restored_img.shape[0]

        ds.save_as(filename, write_like_original=False)


if __name__ == '__main__':

    image = io.imread("SADDLE_PE.JPG", as_gray=True)
    s = Scanner(180, 3 * np.pi / 2, max(image.shape[0]/2, image.shape[1]/2), 180)
    s.load_image(image)
    s.gen_filter(21)
    s.genSpect(180)

    io.imshow(s.get_spect())
    plt.show()

    io.imshow(s.get_restored_img())
    plt.show()

    s.save_to_dicom("nowy.dcm", "PRZYPASNIAK^Michal", "male", "22", "2137", datetime.datetime.now(), "Good Job Team")