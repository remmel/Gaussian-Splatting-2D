import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageViewerCv:
    def __init__(self, window_name='Image Viewer', window_size=(1024, 512)):
        self.window_name = window_name
        self.window_size = window_size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)

    def tensor_to_numpy(self, tensor):
        return (tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

    def show_training_progress(self, predicted, target, epoch):
        predicted_np = cv2.cvtColor(self.tensor_to_numpy(predicted), cv2.COLOR_RGB2BGR)
        target_np = cv2.cvtColor(self.tensor_to_numpy(target), cv2.COLOR_RGB2BGR)

        combined_img = np.hstack((predicted_np, target_np))
        cv2.putText(combined_img, f'Epoch {epoch}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(self.window_name, combined_img)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyAllWindows()


class ImageViewerPlt:
    def __init__(self, window_name='Image Viewer', figsize=(10, 5)):
        self.window_name = window_name
        self.figsize = figsize
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=self.figsize)
        self.fig.canvas.manager.set_window_title(self.window_name)
        self.im1 = None
        self.im2 = None
        plt.ion()  # Turn on interactive mode
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.closed = False

    def on_close(self, event):
        self.closed = True

    def tensor_to_numpy(self, tensor):
        return tensor.permute(1, 2, 0).cpu().detach().numpy()

    def show_training_progress(self, predicted, target, epoch, wait=False):
        predicted_np = self.tensor_to_numpy(predicted)
        target_np = self.tensor_to_numpy(target)

        if self.im1 is None:
            self.im1 = self.ax1.imshow(predicted_np)
            self.ax1.set_title('Predicted')
            self.ax1.axis('off')
        else:
            self.im1.set_data(predicted_np)

        if self.im2 is None:
            self.im2 = self.ax2.imshow(target_np)
            self.ax2.set_title('Target')
            self.ax2.axis('off')

        self.fig.suptitle(f'Epoch {epoch}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if wait:
            plt.waitforbuttonpress()
        else:
            plt.waitforbuttonpress(timeout=0.1)

        return self.closed

    def close(self):
        plt.close(self.fig)