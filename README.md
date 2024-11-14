# Predict Knee Osteoarthritis with Pre-trained ResNet

This uses pre-trained ResNet-18 model, to analyze and assess the severity of knee X-ray images in Osteoarthritis Initiative (OAI) dataset. It aimed to assist the doctor to predict severity of knee osteoarthritis on a scale of 0-4.

Note: The final decision will be on doctor. It is just designed to augment the doctor's decision and fasten the entire process.

Evaluation metrics:
It achieves an accuracy of 74%. The recall rate for each class in this table is as follows:
Class 0: 65% (0.65)
Class 1: 78% (0.78)
Class 2: 72% (0.72)
Class 3: 78% (0.78)
Class 4: 86% (0.86)

There is also an interface available for the doctor to upload the x-ray image and get the predicted class(Class 0-4) with confidence score.
