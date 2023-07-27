from deepface import DeepFace

def predictAge(img):
    objs = DeepFace.analyze(img_path=img,actions=('age',), enforce_detection=False)
    return objs['age']

def predictGender(img):
    objs = DeepFace.analyze(img_path=img,actions=('gender',), enforce_detection=False)
    return objs['gender']

def predictRace(img):
    objs = DeepFace.analyze(img_path=img,actions=('race',), enforce_detection=False)
    return objs['dominant_race']

def predictEmotion(img):
    objs = DeepFace.analyze(img_path=img,actions=('emotion',), enforce_detection=False)
    return objs['dominant_emotion']

