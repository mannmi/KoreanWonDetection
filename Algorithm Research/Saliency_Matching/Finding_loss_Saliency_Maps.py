import cv2


imgpath = r'Content Image.jpg'
image = cv2.imread(imgpath)
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.waitKey(0)
cv2.destroyAllWindows()


imgpath = r'Content Image.jpg'
image = cv2.imread(imgpath)
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

# Set threshold for saliency map
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
# cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)