from ultralytics import YOLO
import json

IMAGE_PATH = "images/1"
JSON_NAME = "57.json"

CLASS_NAME = ['Bird', 'Cloud', 'Flower', 'House', 'Person', 'Sun', 'Tree']

EPSILON = 5

def main():
    # Load your trained model
    model = YOLO("trained-model/best.pt")

    # Run prediction on an image or directory
    results = model.predict(source=IMAGE_PATH, save=True)

    images_loader = ImagesLoader(results)
    process_results = images_loader.get_process_results()

    metadata_loader = MetadataLoader(JSON_NAME)
    metadata = metadata_loader.get_metadata()

    order_object_detection = OrderObjectDetection(process_results, metadata)
    order = order_object_detection.get_order()
    
    objects_list = []

    for r in process_results[0]:
        if r["id"] in order :
            order_id = order.index(r["id"])
            objects_list.append({"ID":r["id"],"cls_ID":r["cls_id"],"cls_name":CLASS_NAME[r["cls_id"]],"Order":order_id})

    print(order)
    print(objects_list)

class ImagesLoader:
    def __init__(self, results):
        self.results = results
        self.process_results = []
    def get_process_results(self):
        self.process_results = []
        # Loop through results
        for r in self.results:
            process_r = []
            print("\n--- Detection Result ---")
            for id, box in enumerate(r.boxes):
                #print(f"cls shape: {box.xyxy}")
                cls_id = int(box.cls[0])                # class id
                conf = float(box.conf[0])               # confidence
                xyxy = box.xyxy[0].tolist()             # [x1, y1, x2, y2]
                process_r.append({"id":id,"cls_id":cls_id ,"xyxy":xyxy})
                print(f"ID: {id}, Class ID: {cls_id}, Confidence: {conf:.2f}, Box: {xyxy}")
            self.process_results.append(process_r)
        return self.process_results

class MetadataLoader:
    def __init__(self,name):
        self.name = name

    def get_metadata(self):
        with open(f"{IMAGE_PATH}/{self.name}", 'r', encoding='utf-8') as file:
            # Load the JSON data
            self.data = json.load(file)
        return self.data

class OrderObjectDetection:
    def __init__(self, process_results, metadata):
        self.process_results = process_results[0]
        self.metadata = metadata
        self.num_objects = len(self.process_results)
        self.order = []

    def get_order(self):
        reverse_actions = self.metadata
        reverse_actions.reverse()

        for action in reverse_actions:
            
            if  self.check_action_points(action):
                reverse_points = action["points"]
                reverse_points.reverse()
                for point in reverse_points:
                    self.update_order(point)
                    if self.num_objects == len(self.order):
                        print("break")
                        self.order.reverse()
                        return self.order
        
    def update_order(self, point):
        point_in_objects = []
        for object in self.process_results:
            if self.check_cls(point, object) and self.check_new_object(object) :
                point_in_objects.append(object["id"])
                #self.order.append(object["id"])
        if len(point_in_objects) == 1:
            self.order.append(point_in_objects[0])
        elif len(point_in_objects) > 1:
            state = False
            for id1 in point_in_objects:
                state = True
                for id2 in point_in_objects:
                    if id1 != id2:
                        if not self.object_1_in_2(id1,id2):
                            state = False

                            break
                if state == True:
                    self.order.append(id1)
                    break

    def check_new_object(self, object):
        for id in self.order:
            if id == object["id"]:
                return False
        return True

    def check_cls(self, point, object):
        if object["xyxy"][0] <= point["x"] and object["xyxy"][1] <= point["y"] and object["xyxy"][2] >= point["x"] and object["xyxy"][3] >= point["y"]:
            return True

    def check_action_points(self,action):
        for key in action.keys():
            if key == "points":
                return True
        return False

    def object_1_in_2(self, id1, id2):
        if (self.process_results[id1]["xyxy"][0] >= self.process_results[id2]["xyxy"][0] - EPSILON and 
            self.process_results[id1]["xyxy"][1] >= self.process_results[id2]["xyxy"][1] - EPSILON  and 
            self.process_results[id1]["xyxy"][2] <= self.process_results[id2]["xyxy"][2] + EPSILON and 
            self.process_results[id1]["xyxy"][3] <= self.process_results[id2]["xyxy"][3] + EPSILON ):
            
            return True
            
if __name__ == "__main__":
    main()