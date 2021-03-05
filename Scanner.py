import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.draw as draw

class Scanner:

    def __init__(self, nDet: int, phi: float, r: float, noViews: int):
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
        self.__restored_img = np.zeros(img.shape)

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
        perc = np.percentile(flat, [3, 98])
        np.clip(self.__restored_img, perc[0], perc[1], self.__restored_img)
        flat = self.__restored_img.flatten()
        self.__restored_img /= max(flat)

        np.clip(self.__restored_img, 0, max(flat), self.__restored_img)


    def genSpect(self, no_views: int):
        self.__spect = []
        spect = []

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

                view.append(cumBrightness / noElements)

            view = np.convolve(view, self.__filter, mode='same')
            self.__reconstruct_view(lines, view)
            spect.append(view)

        self.__spect = np.array(spect)
        self.__norm_img()


if __name__ == '__main__':

    image = io.imread("Kropka.jpg", as_gray=True)
    s = Scanner(90, np.pi, max(image.shape[0]/2, image.shape[1]/2), 90)
    s.load_image(image)
    s.gen_filter(21)
    s.genSpect(90)

    io.imshow(s.get_spect())
    plt.show()

    io.imshow(s.get_restored_img())
    plt.show()