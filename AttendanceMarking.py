
from PIL import Image  # for interacting with images
import numpy as np     # for mathematical operations on arrays
import os              # for interacting with the OS
import cv2


folder = "AttendanceMarkingData"



def load_images(folder):
    R_test = [] # for storing images as rgb, eventually will contain image vectors 
    G_test = []
    B_test = []

    for filename in os.listdir(folder): # is a loop that lists all the files in the directory
            img_path = os.path.join(folder, filename) # generates absolute image paths for the images inside
            if os.path.isfile(img_path):  # Ensure it's a file
                try:
                    with Image.open(img_path) as img: # try catch to open images safely
                        img = img.resize((64, 64)) # resizing to make our life easier
                        img_array = np.array(img) / 255# to convert the image into an array with rgb values

                        R = img_array[:, :, 0].flatten()
                        G = img_array[:, :, 1].flatten()
                        B = img_array[:, :, 2].flatten()
                        
                        R_test.extend(R)
                        G_test.extend(G)
                        B_test.extend(B)

                    # Concatenate R, G, B vectors
                       # img_vector = np.concatenate((R, G, B))
                        #images.append(img_vector) # pushing img vectors to the list
                except IOError:
                    print(f"The image with name: {filename}, with the path: {img_path} isnt correct or the file format isn't correct")
    return np.array(R_test), np.array(G_test), np.array(B_test)

R_test,G_test,B_test = load_images(folder)





def capture_image():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Set resolution to 640x480 as an example
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    
    cv2.namedWindow("Press 'q' to capture", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Press \'q\' to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                frame_rgb = frame_rgb[y:y+h, x:x+w]
            else:
                print("No face detected. Try again.")

                continue
            
            cap.release()
            cv2.destroyAllWindows()
            return frame_rgb

    cap.release()
    cv2.destroyAllWindows()
    return None





def prepare_image(frame_rgb):
    R_real = []
    G_real = []
    B_real = []
    img_array = np.array(frame_rgb)  # Convert to NumPy array
    img = Image.fromarray(img_array)  # Convert image from array to PIL Image
    img = img.resize((64, 64))    # Resize image to match your KNN input size
    img_array = np.array(img) /255    # Convert back to array if needed

    # Extract R, G, B channels
    
    R = img_array[:,:,0].flatten()
    G = img_array[:,:,1].flatten()
    B = img_array[:,:,2].flatten()

    R_real.extend(R)
    G_real.extend(G)
    B_real.extend(B)

    # Concatenate R, G, B vectors
    #img_vector = np.concatenate((R, G, B))
    return np.array(R_real), np.array(G_real), np.array(B_real)






def nearest_neighbour(R, G, B, R_test, G_test, B_test, num_neighbors=5):
    distances = []
    indices = []

    for i in range(len(R_test)):
        distance_R, distance_G, distance_B = euclidean_distance(R_test[i], G_test[i], B_test[i], R, G, B)

        distance = distance_R + distance_G + distance_B
        distances.append(distance)
        indices.append(i)

    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])

    closest_distances = distances[:num_neighbors]
    closest_indices = indices[:num_neighbors]

    return closest_distances, closest_indices


def classify_person(R_test,G_test,B_test,R,G,B, num_neighbors=5, threshold=240):
    print("Database size:")
    closest_distances, closest_indices = nearest_neighbour(R,G,B, R_test,G_test,B_test, num_neighbors)

    print("Closest distances:", closest_distances)
    print("Closest indices:", closest_indices)

    # Check if the smallest distance is less than the threshold
    if min(closest_distances) < threshold:
        return "Person is in the database.", closest_indices
    else:
        return "Person not in the database.", None


def euclidean_distance(R_test, G_test, B_test, R_real, G_real, B_real):
    R_test_norm = R_test / 255
    G_test_norm = G_test / 255
    B_test_norm = B_test / 255

    euclidean_dist_R = np.sqrt(np.sum((R_test_norm - R_real) ** 2))
    euclidean_dist_G = np.sqrt(np.sum((G_test_norm - G_real) ** 2))
    euclidean_dist_B = np.sqrt(np.sum((B_test_norm - B_real) ** 2))

    return euclidean_dist_R, euclidean_dist_G, euclidean_dist_B


def main():
    R,G,B = load_images(folder)
    test_image = capture_image()
    if test_image is not None:
        R_real = test_image[:, :, 0].flatten()
        G_real = test_image[:, :, 1].flatten()
        B_real = test_image[:, :, 2].flatten()

        result, indices = classify_person(R_real, G_real, B_real, R,G,B)
        print(result, indices)

if __name__ == "__main__":
    main()

    
