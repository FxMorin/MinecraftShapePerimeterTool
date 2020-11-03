from glob import glob
import numpy as np
import cv2 as cv
import math

noAccuracyMode = False #Will only do the first grid. Everything will be axis-aligned
verbose = True

GroupingAccuracy = 4 #Recommended 1-6, Grouping accuracy is the limit of gray pixels which need to be together. Higher grouping accuracy will result in less shots and lower accuracy

Steps = 4 #Recommended 1-5, 1 means it steps every pixel. 5 means its steps 5 pixels at a time. Lower numbers will result in higher accuracy

#BrutePower directly corilates to the amount of time it will take for this system to run
BrutePower = 1 #How many times will the algorithm attempt to run the equation again. Max: 25

for fn in glob('./input/*'):
	name = fn.split('/')[-1].split('.')[0].split('\\')[1]
	img = cv.imread(fn)
	img = cv.resize(img, (528,528), interpolation=cv.INTER_CUBIC)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	blk = np.zeros(img.shape, np.uint8)
	totalPixels = np.sum(gray < 230)
	height,width = gray.shape
	cv.imwrite(f'./output/{name}.mask.png', gray)
	squares = []
	w = 0
	h = 0
	notDone = True
	gray[np.where((gray==0))] = 2 #remove the color black since we will be using it as a placeholder
	if verbose:
		print("Getting Points")
	while notDone:
		fine = True
		for ww in range(7):
			for hh in range(7):
				if (h+hh >= height or w+ww >= width) or gray[h+hh,w+ww] >= 230: #If fits into image border
					fine = False
					break
			if not fine:
				break
		if fine:
			squares.append((w,h))
			gray[h:h+7,w:w+7] = 0
			if w >= width:
				h += 7
				w = 0
			else:
				w += 7
		else:
			w += 1
		if w >= width:
			w = 0
			if h >= height:
				notDone = False
			else:
				h += 7
		if w >= width and h >= height:
			notDone = False
	if verbose:
		print("Starting Secondary Algorithm")
	if BrutePower > 25:
		BrutePower = 25
	#Start of the secondary algorithm
	if not noAccuracyMode:
		for x in range(BrutePower):
			notDone = True
			w = 0
			h = 0
			lastH = 0
			while notDone: #Do it again but with seconary algorithm
				if verbose and lastH != math.floor(h/20):
					lastH = math.floor(h/20)
					print((((h/height)*100)*(x+1))/BrutePower)
				hh = 0
				ww = 0
				cg = 0 #Yes, I am using the single most amazing algorithm known to man... Bruteforce
				for Woff in range(-8,6): #Width offset        #default:-9,7
					for Hoff in range(-8,6): #Height offset   #default:-9,7
						countGray = 0
						if (w+Woff-7 >= 0 and h+Hoff+7 < height):
							area = gray[h+Hoff:h+Hoff+7,w+Woff:w+Woff+7]
							if np.sum(area >= 230) == 0:
								countGray = np.count_nonzero(area)
								if countGray > cg:
									ww = w+Woff
									hh = h+Hoff
									cg = countGray
				if cg > GroupingAccuracy*3:                                    #default:14
					squares.append((ww,hh))
					gray[hh:hh+7,ww:ww+7] = 0
					w += 5                                     #default:5
				else:
					w += Steps                                     #default:5
					if w > width:
						w = 0
						if h >= height:
							notDone = False
						else:
							h += 5                             #default:5
				if w > width and h > height:
					notDone = False
	if verbose:
		print("Now adding squares")
	for sqr in squares:
		cv.rectangle(blk, (sqr[0], sqr[1],7,7), (0,255,0), 1)
	if verbose:
		print("Now adding points")
	for sqr in squares:
		cv.rectangle(blk, (sqr[0]+3, sqr[1]+3,1,1), (0,0,255), 1)
	if verbose:
		print("PR0CESSING images")
	out = cv.addWeighted(img, 1.0, blk, 0.25, 1)
	cv.imwrite(f'./output/{name}.png', out)
	shade = cv.addWeighted(blk, 1.0, img, 0.25, 1)
	cv.imwrite(f'./output/{name}.shade.png', shade)
	cv.imwrite(f'./output/{name}.blk.png', blk)
	cv.imwrite(f'./output/{name}.gray.png', gray)
	#print(len(squares))
	# print([ (round(x+(w/2)), round(y+(h/2))) for (x, y, w, h) in squares ])
	blocksMissed = np.sum(gray < 230) - np.sum(gray == 0)
	if verbose:
		print("==================================================")
		print("File Name: "+name)
		print("Total Blocks: "+str(totalPixels))
		print("Blocks Missed: "+str(blocksMissed))
		print("Tnt Used: "+str(len(squares)))
		print("Effeciency: "+str(((totalPixels-blocksMissed)/totalPixels)*100))
		print("==================================================")
