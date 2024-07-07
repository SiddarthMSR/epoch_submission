import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps

# CNN architecture (took help of ChatGPT to adjust parameters)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = Net()
model.load_state_dict(torch.load('alphabet_recognition_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Since the model was trained with 28 x 28 images.
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

extracted_text=[]
'''
Some reasoning behind this function
The hardest part in the task for me was to isolate the characters from images
I found out that the characters are uniform in the test images, like in a grid, that would make this part easier
But that will fail in a general case where the characters are not distributed uniformly so I tried using CV.
Now after identifying the ROIs I was trying to make the isolated images be in the correct order
For this I need to sort w.r.t y-cord first then x. But the sizes of some chars are smaller which means the smaller chars in the first row are considered to be in the second row.
To avoid this I need to group the rows first, I did this by grouping all the chars whose y-diff <= certain_threshold in the same row.
'''
def group_rows(contours, max_y_diff=10):
    rows = []
    current_row = []
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1]) #sort w.r.t y first.
    first_y = cv2.boundingRect(contours[0])[1] 
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if abs(y - first_y) <= max_y_diff:
            current_row.append((x, y, w, h, contour))
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [(x, y, w, h, contour)]
            first_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))
    return rows
'''
So the algo here is sort contours w.r.t y first. Consider the first element. Now this definetly belongs to first row.
Keep moving forward as long as the diff in y-cord is < certain_thresh (avg char height) and all these belong to first row.
After grouping a row we sort it w.r.t x.
Now once we cross this thresh, just repeat the process with this contour.
'''

def print_text(image_name):
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reference for the CV stuff that helped me identify characters: https://stackoverflow.com/questions/46971769/how-to-extract-only-characters-from-image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows = group_rows(contours)
    char_images = []

    for row in rows:
        for x, y, w, h, contour in row:
            roi = img[y:y + h, x:x + w]
            area = w * h
            # Filtering ROIs that are likely to be characters.
            if 150 < area < 1000:
                # Since the ROI is very cropped, this may disrupt the way the model behaves so I added a bit of black border
                border_size = 4
                roi_with_border = cv2.copyMakeBorder(roi, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                char_images.append((x, y, roi_with_border))

    text = ''
    prev_x = 0
    prev_y = 0
    prev_w = 0
    row_start = False  

    for i, (x, y, image) in enumerate(char_images):
        pil_image = Image.fromarray(image)
        pil_image = ImageOps.expand(pil_image, border=4, fill='black')  # For improved accuracy.
        image_tensor = transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        # Feeding the model.
        with torch.no_grad():
            output = model(image_tensor)
            _, chars = torch.max(output, 1)
            char = chr(chars.item() + ord('A'))

        '''
        Below I tried to handle spacing. I added a space if the current_x is sufficiently far from prev_x + prev_width.
        The threshold to add a space seems to be around 25-30 for the test images.
        It doesn't work perfectly because if we start a new row we can't say if we need to add space or not.
        So, I made some assumptions and tried to improve this. 
        If the character is the first in its row check if there is space between (the char and left boundary) or (prev char and right boundary), then add a space.
        '''
        if i > 0: #For the first char theres never a space.
            y_diff = abs(y - prev_y)
            x_diff = abs(x - (prev_x + prev_w))
            # Check if the character is the first in a new row
            row_start = False
            if y_diff > 10:
                row_start = True 
            if row_start:
                # Check if there is space from the left boundary or from the previous character to the right boundary
                if (x > 15 and text) or (prev_x + prev_w < img.shape[1] - 15):
                    text += ' '
            elif x_diff >= 25:
                text += ' '
        text += char
        prev_x = x
        prev_y = y
        prev_w = image.shape[1]
    extracted_text.append(text)
    if __name__=='__main__': #since we are exporting extracted_text, we shouldnt print this when we're running sentiment.py
        print(f"Digital text the model recognized in {image_name}:")
        print(text)

def main():
    for i in range(1, 7):
        image_name = f'line_{i}.png'
        print_text(image_name)


main()
