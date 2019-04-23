import numpy as np


def trimDoubleZeros(ann):
    i = ann.size
    while ann[i-1] == 0 and i > 0:
        if abs(ann[i-1] + ann[i-2]) == 0:
             ann = np.delete(ann, i-1)
             ann = np.delete(ann, i-2)
             i = i - 1;
        i = i - 1;


def polygon_area(x,y):
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def polyarea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def bb_intersection_over_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(annPoly_x[1], trackedPoly_x[0])
	yA = max(annPoly_y[1], trackedPoly_y[0])
	xB = min(annPoly_x[3], trackedPoly_x[2])
	yB = min(annPoly_y[3], trackedPoly_y[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA ) * max(0, yB - yA)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1] ) * (annPoly_y[3] - annPoly_y[1] )
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0] ) * (trackedPoly_y[2] - trackedPoly_y[0] )
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


def bb_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(annPoly_x[1], trackedPoly_x[0])
	yA = max(annPoly_y[1], trackedPoly_y[0])
	xB = min(annPoly_x[3], trackedPoly_x[2])
	yB = min(annPoly_y[3], trackedPoly_y[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA ) * max(0, yB - yA)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1] ) * (annPoly_y[3] - annPoly_y[1] )
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0] ) * (trackedPoly_y[2] - trackedPoly_y[0] )
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	union =  float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return union

def bb_intersection_area(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(annPoly_x[1], trackedPoly_x[0])
    yA = max(annPoly_y[1], trackedPoly_y[0])
    xB = min(annPoly_x[3], trackedPoly_x[2])
    yB = min(annPoly_y[3], trackedPoly_y[2])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea

def midpoint_ann(ptA, ptB):
	return ((ptA[3] + ptA[1]) * 0.5), ((ptB[3] + ptB[1]) * 0.5)

def midpoint_output(ptA, ptB):
	return ((ptA[2] + ptA[0]) * 0.5), ((ptB[2] + ptB[0]) * 0.5)


  
def OTP(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, union, sharedArea):
    if ((annPoly_x[1] == trackedPoly_x[0]) and (annPoly_y[1] == trackedPoly_y[0]) and (annPoly_x[3] == trackedPoly_x[2]) and (annPoly_y[3] == trackedPoly_y[2])):
        otp_errors.append(abs(sharedArea)/abs(union))

centroid_normalized_distance = []

def deviation(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    if ((annPoly_x[1] == trackedPoly_x[0]) and (annPoly_y[1] == trackedPoly_y[0]) and (annPoly_x[3] == trackedPoly_x[2]) and (annPoly_y[3] == trackedPoly_y[2])):
            
        midpoint_ann_x, midpoint_ann_y = midpoint_ann(annPoly_x, annPoly_y)
        #midpoint_ann[i][1] = midpoint_ann_x
        #midpoint_ann[i][2] = midpoint_ann_y
        
        midpoint_output_x, midpoint_output_y = midpoint_output(trackedPoly_x, trackedPoly_y)
        #midpoint_output[i][1] = midpoint_output_x
        #midpoint_output[i][2] = midpoint_output_y
        centroid_normalized_distance_x = np.square(midpoint_output_x - midpoint_ann_x)
        centroid_normalized_distance_y = np.square(midpoint_output_y - midpoint_ann_y)
        centroid_normalized_distance.append(np.sqrt(centroid_normalized_distance_x + centroid_normalized_distance_y))
    
    
def l1_norm_distance(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    midpoint_ann_x, midpoint_ann_y = midpoint_ann(annPoly_x, annPoly_y)
    midpoint_output_x, midpoint_output_y = midpoint_output(trackedPoly_x, trackedPoly_y)
    l1_norm_distance_x = midpoint_ann_x -  midpoint_output_x
    l1_norm_distance_y = midpoint_ann_y -  midpoint_output_y
    
    return l1_norm_distance_x+l1_norm_distance_y
    
def th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    width_ann = annPoly_x[3] - annPoly_x[1]
    height_ann = annPoly_y[3] - annPoly_y[1]
    
    width_output = trackedPoly_x[2] - trackedPoly_x[0]
    height_output = trackedPoly_y[2] - trackedPoly_y[0]
    
    th = (width_ann + height_ann + width_output + height_output)/2
    return th

pbm_list = []
def PBM(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    
    
    if ((annPoly_x[1] == trackedPoly_x[0]) or (annPoly_y[1] == trackedPoly_y[0]) or (annPoly_x[3] == trackedPoly_x[2]) or (annPoly_y[3] == trackedPoly_y[2])):
        distance = l1_norm_distance(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        th_value = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        
        formula = 1-(distance/th_value)
        pbm_list.append(formula)
    else:
        distance = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        th_value = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        formula = 1-(distance/th_value)
        pbm_list.append(formula)
    


outputFile = np.loadtxt("01-Light_video00001.txt", dtype='float', delimiter=' ')
#print(outputFile)

outputFile[:, 4] = outputFile[:, 2] + outputFile[:, 4]
outputFile[:, 3] = outputFile[:, 1] + outputFile[:, 3]


ann_File = np.loadtxt("01-Light_video00001.ann", dtype='float', delimiter=' ')

NoAnnFrames = ann_File[:, 0].size
frameIndex = outputFile[: ,0]
frameIndex = frameIndex.astype(int)
w, h = 2, NoAnnFrames;
errors = [ [0 for x in range( w )] for y in range( h ) ] 
    
errors = np.array(errors ,dtype="float")

w_1, h_1 = 3, NoAnnFrames
area_based_errors = [ [0 for x in range( w_1 )] for y in range( h_1 ) ] 
area_based_errors = np.array(area_based_errors ,dtype="float")


w_2, h_2 = 3, NoAnnFrames
ata_errors = [ [0 for x in range( w_2 )] for y in range( h_2 ) ] 
ata_errors = np.array(ata_errors ,dtype="float")

midpoint = [ [0 for x in range( w_2 )] for y in range( h_2 ) ] 
midpoint = np.array(midpoint,dtype="float")



    
otp_errors = []

#print(errors)
#print('----------------------------------------')
fscores = 0;
for i in range(NoAnnFrames):
    #print(i)
    
    ann = ann_File[i, :]
    #ann = trimDoubleZeros(ann)
    frameId = int(ann[0])
    annPoly_x = ann[1::2]
    annPoly_y = ann[2::2]
    
    # find the corresponding frame in the trackingResultFile
    corrFrameId = frameIndex[np.where(frameIndex == frameId)]
    if corrFrameId.size == 0:
        #errors(i,1) = frameId;
        #errors(i,2) = NaN;
        print('not found')
        
    trackedPos = outputFile[corrFrameId-1, 1:5]
    
    trackedPoly_x = [trackedPos[0][0], trackedPos[0][2], trackedPos[0][2], trackedPos[0][0]]
    trackedPoly_y = [trackedPos[0][1], trackedPos[0][1], trackedPos[0][3], trackedPos[0][3]]
    
    #compute the overlapping area
    
    annArea = polygon_area(annPoly_x, annPoly_y);
    
    trackedArea = polyarea(trackedPoly_x, trackedPoly_y);
    
    sharedArea = bb_intersection_area(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    
    union = bb_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    iou = bb_intersection_over_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    #print(iou)
  
    
    
    errors[i][0] = frameId
    
    errors[i][1] = sharedArea / (annArea + trackedArea - sharedArea)
    
    
    area_based_errors[i][0] = frameId
    area_based_errors[i][1] = sharedArea / trackedArea #p_i
    area_based_errors[i][2] = sharedArea / annArea #r_i
    
    ata_errors[i][0] = frameId
    ata_errors[i][1] = sharedArea
    ata_errors[i][2] = union
    
    midpoint[i][0] = frameId
    midpoint_x, midpoint_y = midpoint_ann(annPoly_x, annPoly_y)
    midpoint[i][1] = midpoint_x
    midpoint[i][2] = midpoint_y
    
    
    
    OTP(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, union, sharedArea)    
    deviation(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    PBM(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    
    #print(errors)
    #print('----------------------------------------')
    
    
#FSCORE 
theta = 0.5
overlaps = errors[:, 1]
#errors = sharedArea / (annArea + trackedArea - sharedArea)
#overlaps = errors
FN = np.count_nonzero(np.isnan(overlaps)) #no track box is ass. with a gt
TP = np.count_nonzero(overlaps >= theta)
FP = np.count_nonzero(overlaps < theta) # a track box is not ass. with a gt
FN = FN + FP # cvpr12 formula
P = TP / (TP + FP)
R = TP / (TP + FN)
fscore = (2*P*R)/(P+R);
#fscores += fscore
    
#fscores = fscores/NoAnnFrames
print('Fscore: ' + str(fscore))


p_i = area_based_errors[:, 1]
r_i = area_based_errors[:, 2]

#AREA BASED F1
p_r_mul = p_i * r_i
p_r_sum = p_i + r_i
p_r_2 = 2 * (p_r_mul/p_r_sum)
p_r_2_sum = np.sum(p_r_2)
area_based_f1 = (1/NoAnnFrames) * p_r_2_sum
print('Area Based F-score: ' + str(area_based_f1))

#OTA
ota = 1 - ((FN+FP)/NoAnnFrames)
print('OTA: '+ str(ota))

Ms = 0

#OTP
if (len(otp_errors) >= 1):
    sum_otp_errors = np.sum(otp_errors)
    Ms = len(otp_errors)
    OTP = (1/abs(Ms)) * sum_otp_errors
    print ("OTP: " + str(OTP))

#ATA
intersection = ata_errors[:, 1]
union = ata_errors[:, 2]
intersection_by_union = abs(intersection)/abs(union )
intersection_by_union_sum = np.sum(intersection_by_union)
ata = (1/NoAnnFrames) * intersection_by_union_sum
print('ATA: ' + str(ata))

#Deviation
if len(centroid_normalized_distance) >= 1:
        
    deviation = 1 - (sum(centroid_normalized_distance)/abs(Ms))
    print('Deviation: ' + str(deviation))

#PBM
pbm = (1/NoAnnFrames) * sum(pbm_list)
print("PBM: " + str(pbm))
