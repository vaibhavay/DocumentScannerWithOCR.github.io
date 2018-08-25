import numpy as np
import cv2


def orderPoints(pts):

	
	rect=np.zeros((4,2),dtype="float32")     #initialising list of coordinates.
									#rect=[[topRightX,topRightY],[topLeftX,topLeftY],[bottomLeftX,bottomLeftY],[bottomRightX,bottomLeftY]]

	s=pts.sum(axis=1)				#this calculation lies in the concept of np.array that invloves extracting of coordinates from the array


	rect[0]=pts[np.argmin(s)]		#sum of the top left coordinate should me minimum 
	rect[2]=pts[np.argmax(s)]		#sum of the bottom right element should be maximum

	diff=np.diff(pts,axis=1)

	rect[1]=pts[np.argmin(diff)]	#difference of top right coordinate should be minimum
	rect[3]=pts[np.argmax(diff)]	#difference of bottom left coordinate should be maximum

	return rect

def fourPointTransform(image,pts):

	rect=orderPoints(pts)			#Calling the orderPoints function

	(tl,tr,br,bl)=rect				#rect matrix is like : rect[[0,0],[0,0],[0,0],[0,0]]


	#Calculating the max Width so that for the birds eye view we have a width . See the reference image in the folder.

	widthA=np.sqrt(((bl[0]-br[0])**2)+((bl[1]-br[0])**2))
	widthB=np.sqrt(((tl[0]-tr[0])**2)+((tl[1]-tr[0])**2))
	maxWidth=max(int(widthA),int(widthB))

	#Calculating the max Width so that for the birds eye view we have a width . See the reference image in the folder.
	
	heightA=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
	heightB=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
	maxHeight=max(int(heightA),int(heightB))


	#dst is the final coordinates matrix that has the coordinates for the birds eye view of the src .

	dst=np.array([
		[0,0],
		[maxWidth-1,0],
		[maxWidth-1,maxHeight-1],
		[0,maxHeight-1]],dtype="float32")


	#M is a perspective matrix , i.e it contains constants M11, M12 etc , that is the transformation matrix for src and dst . "dst=M*src"
	M=cv2.getPerspectiveTransform(rect,dst)	

	#Warped function involves formula "dst=M*src" and it then transforms image coordinate according to dst . 
	#dst(x,y)=src(M11x+M12y+M13M31x+M32y+M33,M21x+M22y+M23M31x+M32y+M33) is the formula .
	warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))

	return warped


	