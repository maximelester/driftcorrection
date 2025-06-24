# driftcorrection.py

# math
import numpy as np

# display
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import symbols as sy

# related to correction algorithm
import cv2

# related to opening SPM images
# from omicronscala import load as stmload
# from access2thematrix import MtrxData
np.set_printoptions(precision=3)

# related to opening other image fomats
from PIL import Image


class DriftCorrection:

	def __init__(self, path, Lx=None, Ly=None, unit='a'):
		'''
		init function of DriftCorrection object
		'''
		self.path = path
		self.Lx = Lx
		self.Ly = Ly
		self.IsCorrected = False
		self.unit = unit
		self.unit_1 = unit+'-1'
		self.GetUnitScaling()
		self.Open()
		self.Defaults()

	def __repr__(self):
		'''
		print DriftCorrection function
		'''
		precision = 5
		d = self.__dict__
		s = ''
		for key, val in zip(d.keys(), d.values()):
			if ['data','fft', 'data_warp', 'data_warp_show', 'fft_warp', 'fft_warped'].count(key)>0:
				val = f'np.array(shape={val.shape})'
			if type(val) == np.float64:
				val = round(val, precision)
			if type(val) == tuple:
				val = [np.round(v, precision) for v in val]
			L = len(key)
			Ntabs = 3-(L)//8
			space = '\t'*Ntabs
			try:
				if str(val)[0] != '.':
					s+=f'{key}{space}{val}\n'
			except:1
		return s

	def Defaults(self):
		'''
		Default values: display options
		'''
		self.CMAP_REAL = 'gist_heat'
		self.CMAP_FFT = 'binary'
		self.GAMMA = 0.4
		self.COLOR_SCALEBAR = 'black'
		self.WIDTHRATIO = 300
		self.SIZERATIO = 6
		self.VMINVMAXFACTOR = 0.0001

	def GetUnitScaling(self):
		'''
		retrieves a multiplication factor
		depending on the chosen unit
		for proper scaling in image display
		'''
		if self.unit == 'pm':
			self.c = 1e12
		elif self.unit == 'a':
			self.c = 1e10
		elif self.unit == 'nm':
			self.c = 1e9
		elif self.unit == 'um':
			self.c = 1e6
		elif self.unit == 'mm':
			self.c = 1e3
		elif self.unit == 'cm':
			self.c = 1e2
		elif self.unit == 'dm':
			self.c = 1e1
		elif self.unit == 'm':
			self.c = 1e0
		elif self.unit == 'dam':
			self.c = 1e-1
		elif self.unit == 'hm':
			self.c = 1e-2
		elif self.unit == 'km':
			self.c = 1e3

	def Open(self):
		'''
		repertoire function to open various image formats
		strictly SPM: 'spm', 'par', 'mtrx'
		others: 'txt', 'npy', 'jpg'
		'''
		if self.path[-3:] == 'txt':
			self.data = np.loadtxt(self.path)
		elif self.path[-3:] == 'npy':
			self.data = np.load(self.path)
		elif self.path[-3:] == 'spm':
			self.data, self.Lx, self.Ly = self.OpenSPM()
		elif self.path[-3:] == 'par':
			self.data, self.Lx, self.Ly = self.OpenPAR()
		elif self.path[-4:] == 'mtrx':
			self.data, self.Lx, self.Ly = self.OpenMTRX()
		elif self.path[-3:] == 'jpg' or self.path[-4:] == 'jpeg' or self.path[-3:] == 'png':
			self.data = self.OpenJPG()

		self.data = np.rot90(np.rot90(self.data))
		self.data-=np.mean(self.data)
		self.fft = FFT(self.data)
		self.UpdateSizes()

	def OpenSPM(self):
		'''
		Opens an SPM file
		'''
		with open(self.path) as f:
			for i, l in enumerate(f):
				if l == 'end of header\n':
					i_start = i+1
				if l == 'end of experiment\n':
					i_finish = i

		with open(self.path) as f:
			a=f.readlines()
			N = int(a[23])
			M = int(a[24])
			Lx = float(a[27])*self.c # into angstrom
			Ly = float(a[28])*self.c # into angstrom
			a=a[i_start:i_finish]

		for i, datapoint in enumerate(a):
			a[i] = float(datapoint)

		z = np.array(a).reshape(N,M)*self.c # to make it into angstrom
		z = np.flip(z, axis=1)

		return z, Lx, Ly

	def OpenPAR(self):
		'''
		Open .par files (requires .tf0 file in the same root!)
		Somehow the filename of the .par (and .tf0) file require a specific format such as:
		m1_zretracedownwedoct2717_14_182021[49-6]stm_spectroscopystm.par
		(underscore positions are CRUCIAL)
		'''
		f = stmload(self.path)
		z = np.array(f[0].data, dtype='float64')
		
		# remove median value line-wise
		# z = AlignRowsMedian(z)
		# z = AlignPlane(z) ### NEEDS X and Y

		# remove mean (we store it later anyway)
		zMean = np.mean(z)
		z-=zMean
		z = z*405.86/32767 # not sure where this number comes from
		
		N = f[0].attrs['ImageSizeinX']
		M = f[0].attrs['ImageSizeinY']
		Lx = f[0].attrs['FieldXSizeinnm']
		Ly = f[0].attrs['FieldYSizeinnm']

		return z, Lx, Ly

	def OpenMTRX(self):
		'''
		'''
		mtrx_data = MtrxData()
		traces, message = mtrx_data.open(self.path)

		if len(traces) == 0:
			print(f"Error in retrieving data in {path[-30:]}, try with a different file.")
			return

		# assuming the first trace (to update, in order to chose which trace)

		im, message = mtrx_data.select_image(traces[0])
		z = im.data
		Lx = im.width*self.c # to angstrom
		Ly = im.height*self.c			
		M, N = z.shape
		X, Y = np.meshgrid(np.linspace(0, Lx, N), np.linspace(0, Ly, M))
		z = SubtractPlane(z, X, Y)  ### needs X and Y
		z = AlignRowsMedian(z)

		return z, Lx, Ly

	def OpenJPG(self):
		'''
		comment/uncomment for average, r, g, b channels
		'''
		# average r, g, b channels
		data = np.mean(np.array(Image.open(self.path), dtype='float64'),axis=2)
		return data
		# r channel
		# return np.array(Image.open(self.path), dtype='float64')[:,:,0]
		# g channel
		# return np.array(Image.open(self.path), dtype='float64')[:,:,1]
		# b channel
		# return np.array(Image.open(self.path), dtype='float64')[:,:,2]

	def UpdateSizes(self):
		'''
		to be called when size/dimensions is updated
		'''
		self.size = self.Lx, self.Ly
		self.pixels = self.data.T.shape
		self.extent = 0, self.Lx, 0, self.Ly
		self.rextent = GetReciExtent(size=(self.size), pixels=(self.pixels))
		self.Kx =self.rextent[1]-self.rextent[0]
		self.Ky =self.rextent[3]-self.rextent[2]

	def ShowRaw(self, axs=None, show=True):
		'''
		Shows the raw data: real and FFT
		'''
		
		if axs is None:
			fig, axs = plt.subplots(ncols=2, tight_layout=True)

		vmin, vmax = GetVminVmax(self.data, n=self.VMINVMAXFACTOR)
		axs[0].imshow(self.data, extent=self.extent, origin='lower', cmap=self.CMAP_REAL, vmin=vmin, vmax=vmax)
		axs[1].imshow(np.abs(self.fft), extent=self.rextent, origin='lower', cmap=self.CMAP_FFT, norm=PowerNorm(self.GAMMA))

		Scalebar(axs[0], size=int(self.Lx//5), width=self.Ly/100, color=self.COLOR_SCALEBAR, unit=self.unit)
		Scalebar(axs[1], size=round(self.Kx/5,1), width=self.Ky/100, color=self.COLOR_SCALEBAR, unit=self.unit_1)

		axs[0].set_title('Raw Data', fontsize=8)
		axs[1].set_title('FFT of Raw Data', fontsize=8)

		if self.IsCorrected == True:
			ShowVectors(K_raw=self.K_raw, K_the=self.K_the, ax=axs[1])

		if show==True:
			plt.show()
		else:
			return axs

	def SetTargetReal(self, r1, r2, omega, theta):
		'''
		Sets the target lattice using real-space definition
		'''
		self.R1_the = r1*np.array([np.cos(theta), np.sin(theta)])
		self.R2_the = r2*np.array([np.cos(theta+omega), np.sin(theta+omega)])
		self.K1_the, self.K2_the = GetReciVectors(r1, r2, omega, theta)
		self.r1_the, self.r2_the = r1, r2
		self.omega_the, self.theta_the = omega, theta
		self.K_the = self.K1_the, self.K2_the

	def SetTargetReci(self, K1, K2):
		'''
		Sets target lattice using reciprocal space vectors
		'''
		self.K1_the, self.K2_the = np.array(K1), np.array(K2)
		self.r1_the, self.r2_the, self.omega_the, self.theta_the = GetRealVectors(K1, K2)
		self.K_the = self.K1_the, self.K2_the

	def SetRawReal(self, r1, r2, omega, theta):
		'''
		Sets as-observed lattice using real space definition
		'''
		self.R1_raw = r1*np.array([np.cos(theta), np.sin(theta)])
		self.R2_raw = r2*np.array([np.cos(theta+omega), np.sin(theta+omega)])
		self.K1_raw, self.K1_raw = GetReciVectors(r1, r2, omega, theta)
		self.r1_raw, self.r2_raw = r1, r2
		self.omega_raw, self.theta_raw = omega, theta
		self.K_raw = self.K1_raw, self.K2_raw

	def SetRawReci(self, K1, K2):
		'''
		Sets as-observed lattice using reciprocal vectors
		'''
		self.K1_raw, self.K2_raw = np.array(K1), np.array(K2)
		self.r1_raw, self.r2_raw, self.omega_raw, self.theta_raw = GetRealVectors(K1, K2)
		self.K_raw = self.K1_raw, self.K2_raw

	def Transform(self):
		'''
		assumes the data, target and observed lattices are stored properly
		'''

		# Reciprocal matrix (M_reci)

		self.M_reci = GetMatrix(self.K1_raw, self.K2_raw, self.K1_the, self.K2_the)

		# Reciprocal normalized matrix (M_reci_norm)

		self.M_reci_norm, self.M_dict = DecomposeMatrix(self.M_reci)

		# Real matrix (M_real)

		self.M_real = np.linalg.inv(self.M_reci.T)

		# Real normalized matrix (M_real_norm))

		self.M_real_norm, self.M_real_dict = DecomposeMatrix(self.M_real)

		# Perform Warping

		self.data_warp, self.extent_warp, self.fft_warped, self.rextent_warped = TransformImage(self.data, initial_extent=self.extent, matrix=self.M_real)
		
		# Warped real space data (size and pixels)

		self.Lx_warp = self.extent_warp[1]-self.extent_warp[0]
		self.Ly_warp = self.extent_warp[3]-self.extent_warp[2]
		self.N_warp, self.M_warp = self.pixels_warp = self.data_warp.T.shape
		self.data_warp_show = self.data_warp.copy()
		self.data_warp_show[self.data_warp==0]=np.nan

		# FFT of warped data
		# here the FFT is obtained by FFT operation of the warped real space data

		self.fft_warp = FFT(self.data_warp)
		self.rextent_warp = GetReciExtent(size=(self.Lx_warp, self.Ly_warp), pixels=self.pixels_warp)
		self.Kx_warp = self.rextent_warp[1]-self.rextent_warp[0]
		self.Ky_warp = self.rextent_warp[3]-self.rextent_warp[2]

		# Warped raw FFT (≠ to the previous case!!)
		# this time the FFT is obtained from warping the raw FFT
		# the artifacts and size differ, but the information should be the same
		# note the attributes are called self.<attribute>ed

		self.Kx_warped = self.rextent_warped[1]-self.rextent_warped[0]
		self.Ky_warped = self.rextent_warped[3]-self.rextent_warped[2]
		self.N_warped, self.M_warped = self.pixels_warped = self.fft_warped.T.shape


		self.IsCorrected = True

	def ShowAll(self, show=True, kmax=None, scalebars=None, titles=True):
		'''
		Shows all results in a combined plot for convenience
		'''

		# column

		fig, axs = plt.subplots(ncols=3, nrows=2, tight_layout=True, figsize=(12,8))

		self.axs = axs

		# show raw and fft of raw
		vmin, vmax = GetVminVmax(self.data, n=self.VMINVMAXFACTOR)
		axs[0,0].imshow(self.data, origin='lower', extent=self.extent, cmap=self.CMAP_REAL, vmin=vmin, vmax=vmax)
		axs[1,0].imshow(np.abs(self.fft), origin='lower', extent=self.rextent, cmap=self.CMAP_FFT, norm=PowerNorm(self.GAMMA))
		
		# show warp and fft of warp
		# vmin, vmax = GetVminVmax(self.data_warp, n=0.005)
		a=axs[0,1].imshow(self.data_warp_show, origin='lower', extent=self.extent_warp, cmap=self.CMAP_REAL, vmin=vmin, vmax=vmax)
		fig.colorbar(a, ax=axs[0,1], pad=0.02, aspect=30)
		axs[1,1].imshow(np.abs(self.fft_warp), origin='lower', extent=self.rextent_warp, cmap=self.CMAP_FFT, norm=PowerNorm(self.GAMMA))

		# show warped fft
		axs[1,2].imshow(np.abs(self.fft_warped), origin='lower', extent=self.rextent_warped, cmap=self.CMAP_FFT, norm=PowerNorm(self.GAMMA))
		
		# get xlim and ylim

		xmin = min(self.extent[0], self.extent_warp[0])
		xmax = max(self.extent[1], self.extent_warp[1])
		ymin = min(self.extent[2], self.extent_warp[2])
		ymax = max(self.extent[3], self.extent_warp[3])
		
		x = xmax-xmin
		y = ymax-ymin

		# titles
		if titles == True:
			axs[0,0].set_title('Warped Data', fontsize=8)
			axs[1,0].set_title('FFT of Raw Data', fontsize=8)
			axs[0,1].set_title('Warped Raw Data', fontsize=8)
			axs[1,1].set_title('FFT of Warped Data', fontsize=8)
			axs[1,2].set_title('Warped FFT of Raw Data', fontsize=8)

		# get kxlim and kylim

		kxmin = min(self.rextent[0], self.rextent_warp[0], self.rextent_warped[0])
		kxmax = max(self.rextent[1], self.rextent_warp[1], self.rextent_warped[1])
		kymin = min(self.rextent[2], self.rextent_warp[2], self.rextent_warped[2])
		kymax = max(self.rextent[3], self.rextent_warp[3], self.rextent_warped[3])
		
		kx = kxmax-kxmin
		ky = kymax-kymin

		if kmax is not None:
			kxmin, kxmax = -kmax, kmax
			kymin, kymax = -kmax, kmax
			kx = 2*kmax
			ky = 2*kmax

		axs[1,0].set_xlim([kxmin, kxmax])
		axs[1,0].set_ylim([kymin, kymax])

		axs[1,1].set_xlim([kxmin, kxmax])
		axs[1,1].set_ylim([kymin, kymax])

		axs[1,2].set_xlim([kxmin, kxmax])
		axs[1,2].set_ylim([kymin, kymax])

		
		# custom
		if scalebars is None:
			Scalebar(axs[0,0], size=int(x//self.SIZERATIO), width=y/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit)
			Scalebar(axs[0,1], size=int(x//self.SIZERATIO), width=y/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit)
		else:
			Scalebar(axs[0,0], size=scalebars[0], width=y/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit)
			Scalebar(axs[0,1], size=scalebars[1], width=y/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit)
		Scalebar(axs[1,0], size=round(kx/self.SIZERATIO,1), width=ky/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit_1)
		Scalebar(axs[1,1], size=round(kx/self.SIZERATIO,1), width=ky/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit_1)
		Scalebar(axs[1,2], size=round(kx/self.SIZERATIO,1), width=ky/self.WIDTHRATIO, color=self.COLOR_SCALEBAR, unit=self.unit_1)

		# third panel: show info
		axs[0,2].axis('off')

		if show == True:
			plt.show()

	def PrintResults(self):	
		'''
		Prints various results in the console:
		matrices, real, reci, normalized, rotation, scaling, shear parameters,...
		pixel numbers, dimensions, both for real and fft data
		'''
		if self.unit=='a':
			unit = sy.angstrom
			if self.unit_1 == 'a-1':
				unit_1 = sy.angstrom+sy.superscript_minus1
			else:
				unit_1 = self.unit_1
		else:
			unit = self.unit
			unit_1 = self.unit_1

		PrintResults(K_raw=self.K_raw, K_the=self.K_the, data=self.data, extent=self.extent, units=(unit,unit_1))

	def ShowVectors(self, axs, mode='points'):
		'''
		Adds points (or arrows if mode='vectors') onto the specified ax/axs
		'''
		try:
			ShowVectors(K_raw=self.K_raw, K_the=self.K_the, ax=axs, mode=mode)
		except:
			for ax in axs:
				ShowVectors(K_raw=self.K_raw, K_the=self.K_the, ax=ax, mode=mode)

	def SaveData(self, which='both'):
		'''
		Saves data: which can be 'real' (only warped real data), 'fft' (only warped fft)
		or 'both' (both real and warped fft)
		format is npy. Size of image contained in the file name for convenience.
		'''

		if self.IsCorrected==True:

			name = self.path[:self.path.index('.')]
			name+=f'_corrected_(Lx={self.Lx_warp:.3f}_Ly={self.Ly_warp:.3f}).npy'

			if which == 'real' or which == 'both':
				np.save(name, self.data_warp)
				SaveArraySPM(data=self.data_warp, extent=self.extent_warp, name=f'warped_data.spm')

			if which == 'fft' or which == 'both':
				name = name[:-4]+f'_FFTWarp.npy'
				np.save(name, self.fft_warped)






# SPM file

def SaveArraySPM(data, extent, name='data.spm'):

	N, M = data.T.shape
	x0, x1, y0, y1 = extent
	lx, ly = x1-x0, y1-y0
	lx_meter = 1e-10*lx
	ly_meter = 1e-10*ly


	data = data.ravel()



	with open(name, "+x") as f:
		f.write('ISO/TC 201 SPM data transfer format\n')
		f.write('general information\n\n\n\n\n')
		f.write('Created by Drift Correction by Maxime Le Ster (c).  Bogus acquisition parameters.\n')
		f.write('MAP_SC\n')
		f.write('-1\n'*7)
		f.write('scan information\n')
		f.write('REGULAR MAPPING\n')
		f.write('XYZ closed-loop scanner\n')
		f.write('sample XYZ scan\n')
		f.write('X\n')
		f.write('left to right\n')
		f.write('Y\n')
		f.write('top to bottom\n')
		f.write(f'{N}\n')
		f.write(f'{M}\n')
		f.write('m\n')
		f.write('m\n')
		f.write(f'{lx_meter}\n')
		f.write(f'{ly_meter}\n')
		f.write('m\n')
		f.write('m\n')
		f.write('0\n')
		f.write('0\n')
		f.write('0\n')
		f.write('m/s\n')
		f.write('0.0\n')
		f.write('Hz\n')
		f.write('0.0\n\n')
		f.write(f'sample biased\n')
		f.write('0.0\n')
		f.write('0\n')
		f.write(f'\n'*5)
		f.write('environment description\n')
		f.write('software\n')
		f.write('300\n')
		f.write('1.0e5\n')
		f.write('40\n\n')
		f.write(f'probe description\n')
		f.write('software\n\n')
		f.write('0.0\n')
		f.write('0.0\n')
		f.write('0.0\n')
		f.write('0\n')
		f.write('0\n')
		f.write('0\n\n')
		f.write('sample description\n')
		f.write('Z (Forward)\n\n\n')
		f.write('single-channel mapping description\n')
		f.write('Z (Forward)\n')
		f.write('m\n\n')
		f.write('spectroscopy description\n\n')
		f.write('REGULAR\n\n')
		f.write(f'n\n')
		f.write('0.0\n')
		f.write('0.0\n')
		f.write('0.0\n')
		f.write('0.0\n')
		f.write('0\n')
		f.write('0\n\n')
		f.write('n\n')
		f.write('0.0\n\n')
		f.write('data treatment description\n')
		f.write('post-treated data\n\n\n\n\n')
		f.write('multi-channel mapping description\n')
		f.write('1\n')
		f.write('Z (Forward)\n')
		f.write('m\n')
		f.write('Z (Forward)\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('\n')
		f.write('n\n')
		f.write('\n')
		f.write('end of header\n')
		for d in data:
			f.write(f'{d:.7e}\n')
		f.write('end of experiment\n\n')


'''




'''

# Display-only functions

def GetVminVmax(data, n):
	'''
	get the vertical bounds of the data (trimmed by n)
	'''
	M, N = data.shape

	data_ = data.copy()

	data_[np.isnan(data)]=0

	a = np.histogram(data_.ravel(), bins=max(N,M), density=True)
	x = a[1][:-1]
	y = a[0]
	xtrim = x[y>n]
	ytrim = y[y>n]
	if len(xtrim)*len(ytrim)==0:
		return None, None
	xmin = np.min(xtrim)
	xmax = np.max(xtrim)
	return xmin, xmax

def Scalebar(ax, size=None, width=None, color='black', unit='a', fontsize=8, loc='upper right', background=False, alphabg=0.8):
	'''
	adds scalebar of size, and width and color on ax
	'''

	# hide x and y axes

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# size/width if unspecified

	if size is None:
		size_x = ax.get_xlim()[1]-ax.get_xlim()[0]
		size = size_x/5
		if size>1:
			size = int(size)
		else:
			size = round(size,1)
			# size = int(size)
	
	if width is None:
		size_y = ax.get_ylim()[1]-ax.get_ylim()[0]
		width = size_y/140

	# font

	fontprops = fm.FontProperties(size=fontsize, weight='heavy')

	# unit

	if unit == 'pm':
		text = f'{size} pm'
	elif unit == 'a':
		text = f'{size} {sy.angstrom}'
	elif unit == 'nm':
		text = f'{size} nm'
	elif unit == 'um':
		text = f'{size} {sy.mu}m'
	elif unit == 'mm':
		text = f'{size} mm'
	elif unit == 'cm':
		text = f'{size} cm'
	elif unit == 'dm':
		text = f'{size} dm'
	elif unit == 'm':
		text = f'{size} m'
	elif unit == 'dam':
		text = f'{size} dam'
	elif unit == 'hm':
		text = f'{size} hm'
	elif unit == 'km':
		text = f'{size} km'
	
	elif unit == 'pm-1':
		text = f'{size} {sy.angstrom}{sy.superscript_minus1}'
	elif unit == 'a-1':
		text = f'{size} {sy.angstrom}{sy.superscript_minus1}'
	elif unit == 'nm-1':
		text = f'{size} nm{sy.superscript_minus1}'
	elif unit == 'um-1':
		text = f'{size} {sy.mu}m{sy.superscript_minus1}'
	elif unit == 'mm-1':
		text = f'{size} mm{sy.superscript_minus1}'
	elif unit == 'cm-1':
		text = f'{size} cm{sy.superscript_minus1}'
	elif unit == 'dm-1':
		text = f'{size} dm{sy.superscript_minus1}'
	elif unit == 'm-1':
		text = f'{size} m{sy.superscript_minus1}'
	elif unit == 'dam-1':
		text = f'{size} dam{sy.superscript_minus1}'
	elif unit == 'hm-1':
		text = f'{size} hm{sy.superscript_minus1}'
	elif unit == 'km-1':
		text = f'{size} km{sy.superscript_minus1}'

	bar = AnchoredSizeBar(ax.transData, size, text, loc, pad=0.5, sep=5,
		color=color, frameon=False, size_vertical=width, fontproperties=fontprops)
	bar.set(clip_on=True)

	if background==True:

		color_background = np.array([1, 1, 1, 0])-np.array(mpl.colors.to_rgba(color))
		color_background[3] = alphabg
		ax.text(0.78, 0.94, s='              ', bbox={'edgecolor':[0,0,0,0], 'facecolor': color_background}, transform=ax.transAxes, fontsize=10, clip_on=True)
	
	ax.add_artist(bar)

def ShowVectors(K_raw, K_the, show=False, ax=None, mode='points', colors=None):
	if ax is None:
		fig, ax = plt.subplots()

	if colors is None:
		colors = ['black', 'red']
	labels = ['raw', 'the']
	Ks = [K_raw, K_the]

	for i, K in enumerate(Ks):
		if mode == 'vectors':
			ax.arrow(0,0, dx=K[0][0], dy=K[0][1], color=colors[i], label=labels[i], length_includes_head=True)
			ax.arrow(0,0, dx=K[1][0], dy=K[1][1], color=colors[i], length_includes_head=True)
		elif mode == 'points':
			ax.plot(K[0][0], K[0][1], marker='.', lw=0, color=colors[i], label=labels[i])
			ax.plot(K[1][0], K[1][1], marker='.', lw=0, color=colors[i])
		ax.text(K[0][0], K[0][1], s=' 10', color=colors[i], fontsize=8, clip_on=True)
		ax.text(K[1][0], K[1][1], s=' 01', color=colors[i], fontsize=8, clip_on=True)

	ax.set_aspect('equal', 'box')
	ax.legend(loc='lower right')

	if show == True:
		plt.show()

def PrintResults(K_raw, K_the, data, extent, units):

	# get important info

	M = GetMatrix(K_raw[0], K_raw[1], K_the[0], K_the[1])
	K_cor = [M@K_raw[0], M@K_raw[1]]
	Mn, dic = DecomposeMatrix(M)

	Mr = np.linalg.inv(M).T
	Mnr, dicr = DecomposeMatrix(Mr)

	Lx, Ly = extent[1]-extent[0], extent[3]-extent[2]
	size = data.T.shape

	rextent = GetReciExtent(size=(Lx, Ly), pixels=size)
	fft = np.fft.fftshift(np.fft.fft2(data))

	dataWarp, extentWarp, WarpFFT, rextentWarpFFT = TransformImage(data, extent, matrix=Mr)

	FFTWarp = np.fft.fftshift(np.fft.fft2(dataWarp))
	rextentWarp = GetReciExtent(size=(extentWarp[1]-extent[0], extentWarp[3]-extent[2]), pixels=dataWarp.T.shape)

	# print

	print(f'Raw Reciprocal Base Vectors:')
	print(f'K10\t[{K_raw[0][0]:.5f}\t{K_raw[0][1]:.5f}]')
	print(f'K01\t[{K_raw[1][0]:.5f}\t{K_raw[1][1]:.5f}]\n')
	
	print(f'Theoretical Reciprocal Base Vectors:')
	print(f'K10\t[{K_the[0][0]:.5f}\t{K_the[0][1]:.5f}]')
	print(f'K01\t[{K_the[1][0]:.5f}\t{K_the[1][1]:.5f}]\n')
	
	print(f'Corrected Reciprocal Base Vectors:')
	print(f'K10\t[{K_cor[0][0]:.5f}\t{K_cor[0][1]:.5f}]')
	print(f'K01\t[{K_cor[1][0]:.5f}\t{K_cor[1][1]:.5f}]\n')
	
	print(f'Correction Matrix (reciprocal space):')
	print(f'Mreci\t[{M[0,0]:.5f}\t{M[0,1]:.5f}]')
	print(f'     \t[{M[1,0]:.5f}\t{M[1,1]:.5f}]')
	print(f'det\t{np.linalg.det(M):.5f}\n')
	
	print(f'Normalized Correction Matrix (reciprocal space):')
	print(f'Mnorm\t[{Mn[0,0]:.5f}\t{Mn[0,1]:.5f}]')
	print(f'     \t[{Mn[1,0]:.5f}\t{Mn[1,1]:.5f}]')
	print(f'det\t{np.linalg.det(Mn):.5f}\n')
	
	print(f'Correction Matrix components (reciprocal space):')
	print(f'sx\t{dic["sx"]:.5f}')
	print(f'sy\t{dic["sy"]:.5f}')
	print(f'tau\t{dic["tau"]:.5f}')
	print(f'{sy.theta}\t{dic["theta"]:.5f} rad ({dic["theta"]*180/np.pi:.3f}{sy.degree})\n')
	
	print(f'Correction Matrix (real space):')
	print(f'Mreal\t[{Mr[0,0]:.5f}\t{Mr[0,1]:.5f}]')
	print(f'     \t[{Mr[1,0]:.5f}\t{Mr[1,1]:.5f}]')
	print(f'det\t{np.linalg.det(Mr):.5f}\n')
	
	print(f'Normalized Correction Matrix (real space):')
	print(f'Mnorm\t[{Mnr[0,0]:.5f}\t{Mnr[0,1]:.5f}]')
	print(f'     \t[{Mnr[1,0]:.5f}\t{Mnr[1,1]:.5f}]')
	print(f'det\t{np.linalg.det(Mnr):.5f}\n')
	
	print(f'Correction Matrix components (real space):')
	print(f'sx\t{dicr["sx"]:.5f}')
	print(f'sy\t{dicr["sy"]:.5f}')
	print(f'tau\t{dicr["tau"]:.5f}')
	print(f'{sy.theta}\t{dicr["theta"]:.5f} rad ({dicr["theta"]*180/np.pi:.3f}{sy.degree})\n')

	print(f'Initial image:')
	print(f'N {sy.times} M\t({size[0]} {sy.times} {size[1]})')
	print(f'Lx {sy.times} Ly\t({Lx:.5f} {sy.times} {Ly:.5f}) {units[0]}{sy.superscript_2}\n')

	print(f'Final image:')
	print(f'N {sy.times} M\t({dataWarp.shape[1]} {sy.times} {dataWarp.shape[0]})')
	print(f'Lx {sy.times} Ly\t({extentWarp[1]-extentWarp[0]:.5f} {sy.times} {extentWarp[3]-extentWarp[2]:.5f}) {units[1][:-2]}{sy.superscript_2}\n')

	print(f'Initial FFT of the data:')
	print(f'N {sy.times} M\t({fft.T.shape[0]} {sy.times} {fft.T.shape[1]})')
	print(f'kx_min\t{rextent[0]:.5f} {units[1]}')
	print(f'kx_max\t{rextent[1]:.5f} {units[1]}')
	print(f'ky_min\t{rextent[2]:.5f} {units[1]}')
	print(f'ky_max\t{rextent[3]:.5f} {units[1]}')
	print(f'Kx {sy.times} Ky\t{rextent[1]-rextent[0]:.5f} {sy.times} {rextent[3]-rextent[2]:.5f} {units[1][:-2]}{sy.superscript_minus2}')
	print(f'dkx\t{(rextent[1]-rextent[0])/fft.T.shape[0]:.5f} {units[1]}')
	print(f'dky\t{(rextent[3]-rextent[2])/fft.T.shape[1]:.5f} {units[1]}\n')

	print(f'FFT of the warped data:')
	print(f'N {sy.times} M\t({FFTWarp.T.shape[0]} {sy.times} {FFTWarp.T.shape[1]})')
	print(f'kx_min\t{rextentWarp[0]:.5f} {units[1]}')
	print(f'kx_max\t{rextentWarp[1]:.5f} {units[1]}')
	print(f'ky_min\t{rextentWarp[2]:.5f} {units[1]}')
	print(f'ky_max\t{rextentWarp[3]:.5f} {units[1]}')
	print(f'Kx {sy.times} Ky\t{rextentWarp[1]-rextentWarp[0]:.5f} {sy.times} {rextentWarp[3]-rextentWarp[2]:.5f} {units[1][:-2]}{sy.superscript_minus2}')
	print(f'dkx\t{(rextentWarp[1]-rextentWarp[0])/FFTWarp.T.shape[0]:.5f} {units[1]}')
	print(f'dky\t{(rextentWarp[3]-rextentWarp[2])/FFTWarp.T.shape[1]:.5f} {units[1]}\n')

	print(f'Warped FFT data:')
	print(f'N {sy.times} M\t({WarpFFT.T.shape[0]} {sy.times} {WarpFFT.T.shape[1]})')
	print(f'kx_min\t{rextentWarpFFT[0]:.5f} {units[1]}')
	print(f'kx_max\t{rextentWarpFFT[1]:.5f} {units[1]}')
	print(f'ky_min\t{rextentWarpFFT[2]:.5f} {units[1]}')
	print(f'ky_max\t{rextentWarpFFT[3]:.5f} {units[1]}')
	print(f'Kx {sy.times} Ky\t{rextentWarpFFT[1]-rextentWarpFFT[0]:.5f} {sy.times} {rextentWarpFFT[3]-rextentWarpFFT[2]:.5f} {units[1][:-2]}{sy.superscript_minus2}')
	print(f'dkx\t{(rextentWarpFFT[1]-rextentWarpFFT[0])/WarpFFT.T.shape[0]:.5f} {units[1]}')
	print(f'dky\t{(rextentWarpFFT[3]-rextentWarpFFT[2])/WarpFFT.T.shape[1]:.5f} {units[1]}\n')

def Show(z, extent, axs, title=None):

	rextent = GetReciExtent(size=(extent[1]-extent[0], extent[3]-extent[2]), pixels=z.T.shape)

	zshow = z.copy()
	zshow[zshow==0]=np.nan

	vmin, vmax = GetVminVmax(z, n=self.VMINVMAXFACTOR)
	axs[0].imshow(zshow, extent=extent, origin='lower', cmap='gist_heat', vmin=vmin, vmax=vmax)
	axs[1].imshow(np.abs(FFT(z)), extent=rextent, origin='lower', cmap='binary_r', norm=PowerNorm(0.5))

	Scalebar(axs[0], color='white', unit='a')
	Scalebar(axs[1], color='black', unit='a-1')

	if title is not None:
		axs[0].set_title(title, fontsize=8)
		axs[1].set_title(title+f' (FFT)', fontsize=8)

# Crystallography functions/FFT

def GetReciExtent(size, pixels):
	'''
	Returns extent (in [size unit]-1) of FFT
	'''
	Lx, Ly = size
	N, M = pixels

	dx, dy = Lx/N, Ly/M
	dkx, dky = 1/Lx, 1/Ly
	kxmax, kymax = 1/(2*dx), 1/(2*dy)

	if N%2==1:
		Kx = np.linspace(-kxmax, kxmax, N)
	else:
		Kx = np.linspace(-kxmax-dkx/2, kxmax-dkx/2, N)
	if M%2==1:
		Ky = np.linspace(-kymax, kymax, M)
	else:
		Ky = np.linspace(-kymax-dky/2, kymax-dky/2, M)

	extent = np.min(Kx), np.max(Kx), np.min(Ky), np.max(Ky)
	return extent

def FFT(data, remove_mean=True):
	if remove_mean==True:
		av = np.mean(data)
	else:
		av = 0
	return np.fft.fftshift(np.fft.fft2(data-av))

def GetReciVectors(r1, r2, omega, theta):
	'''
	Returns reciprocal vectors K1, K2 for any real space lattice parameters
	if omega = 120º (hexagonal lattice), K1, K2, K1-K2 are returned
	(for symmetry reasons)
	'''

	k1 = 1/r1/np.sin(omega)
	k2 = 1/r2/np.sin(omega)

	K1 = k1*np.sin(omega+theta), -k1*np.cos(omega+theta)
	K2 = k2*-np.sin(theta), k2*np.cos(theta)

	return np.array((K1, K2))

def GetRealVectors(K1, K2):

	K1_, K2_ = K1/np.linalg.norm(K1), K2/np.linalg.norm(K2)

	theta = np.arctan2(-K2_[0], K2_[1])
	omega_plus_theta = np.arctan2(K1_[0], -K1_[1])
	omega = RestrictOmega(omega_plus_theta-theta)

	# print(f'theta: {theta*180/np.pi:.2f}')
	# print(f'omega: {omega*180/np.pi:.2f}')

	r1 = 1/np.sin(omega)/np.linalg.norm(K1)
	r2 = 1/np.sin(omega)/np.linalg.norm(K2)

	# print(f'r1: {r1:.2f}')
	# print(f'r2: {r2:.2f}')

	R1 = r1*np.array([np.cos(theta), np.sin(theta)])
	R2 = r2*np.array([np.cos(theta+omega), np.sin(theta+omega)])

	return r1, r2, omega, theta

def GetRealVectors_old(K1, K2):
	'''
	returns real vectors R1 and R2
	'''
	k1 = np.linalg.norm(K1)
	k2 = np.linalg.norm(K2)

	K1 = k1*np.exp(1j*np.arctan2(K1[1], K1[0]))
	K2 = k2*np.exp(1j*np.arctan2(K2[1], K2[0]))

	
	theta = (np.log(K2/k2)/1j - np.pi/2).real
	omega = np.abs(-1j*np.log((K2/k2)/(K1/k1)))

	r1 = np.abs(1/k1/np.sin(omega))
	r2 = np.abs(1/k2/np.sin(omega))

	return r1, r2, RestrictOmega(omega), RestrictTheta(theta)

def RestrictOmega(omega):
	'''
	restriction function for omega input values (omega in rad)
	because e.g. -3º(87º), 180º(NaN), 0º(NaN), 201º(21º) are wrong (correct) values

	'''
	# if omega==0:
		# omega = np.nan
	# if omega%np.pi==0:
		# omega = np.nan

	return omega%np.pi

def RestrictTheta(theta, epsilon=1e-15):
	'''
	Returns theta in [-pi, pi]
	restriction function for theta input values (theta in rad)
	because e.g. -93º(87º), 180.3º(0.3º), 361º(1º) are wrong (correct) values
	'''
	theta = theta+epsilon

	return (theta+np.pi)%(2*np.pi)-np.pi

def SubtractPlane(Y, X1, X2):
	'''
	returns Y but 1degree-polynomial subtracted
	'''
	X = np.hstack(   ( np.reshape(X1, (X1.shape[0]*X1.shape[1], 1)) , np.reshape(X2, (X2.shape[0]*X2.shape[1], 1)) ) )
	X = np.hstack(   ( np.ones((X1.shape[0]*X1.shape[1], 1)) , X ))
	YY = np.reshape(Y, (X1.shape[0]*X1.shape[1], 1))
	theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
	plane = np.reshape(np.dot(X, theta), (X1.shape[0], X1.shape[1]))
	Y_sub = Y - plane
	return Y_sub

def AlignRowsMedian(z):

	za = z.copy()
	prev_m = np.median(za[0])

	for i in range(len(za)):
		za[i]-=prev_m
		try:
			prev_m = np.median(za[i+1])
		except:
			1
	return za

# Drift compensation functions

def GetMatrix(K1_raw, K2_raw, K1_the, K2_the):
	'''
	Returns the linear matrix M such that
	M @ K1_raw -> K1_the
	M @ K2_raw -> K2_raw

	'''
	k1x, k1y = K1_raw
	k2x, k2y = K2_raw
	k1X, k1Y = K1_the
	k2X, k2Y = K2_the

	a = k1X*k2y-k2X*k1y
	b = k2X*k1x-k1X*k2x
	c = k1Y*k2y-k2Y*k1y
	d = k2Y*k1x-k1Y*k2x

	return 1/(k1x*k2y-k2x*k1y)*np.array([[a, b], [c, d]])

def DecomposeMatrix(M):
	'''
	returns the matrix MN, which is M "stripped off" its scaling components
	named here as a normalized matrix, with the property that
	det(MN) = 1
	this is essential in case of significant scaling error between input and output
	reciprocal lattice vector lengths
	and ensures that the linear transformation of the image preserves (mostly) the pixel count

	also returns a dictionary containing sx, sy, tau and theta
	M = MS @ MT @ MR (scaling, shear, and rotation matrices)
	'''

	a, b, c, d = M[0,0], M[0,1], M[1,0], M[1,1]
	
	theta = np.arctan2(c,d)
	tau = (a*np.sin(theta)+b*np.cos(theta))/(a*np.cos(theta)-b*np.sin(theta))
	sx = a/(np.cos(theta)+tau*np.sin(theta))
	
	if np.isnan(sx):
		sx = b/(tau*np.cos(theta)-np.sin(theta))	
		print(f'sx was np.nan')
	
	sy = c/np.sin(theta)
	if np.isnan(sy):
		sy = d/np.cos(theta)
		print(f'sy was np.nan')

	MS = np.array([[sx, 0],[0, sy]])
	MT = np.array([[1, tau],[0, 1]])
	MR = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

	MN = MT@MR

	return MN, {'sx':sx, 'sy':sy, 'tau':tau, 'theta':theta}

def TransformImage(data, initial_extent, matrix):
	'''
	data is the real space data (STM or otherwise)
	input "matrix" is the real matrix (not normalized)
	initial_extent is the real space extent in the form: xmin, xmax, ymin, ymax

	returns warped real space data, new real space extent, warped FFT data and the FFT extent

	'''
	N, M = data.T.shape
	Lx, Ly = initial_extent[1]-initial_extent[0], initial_extent[3]-initial_extent[2]
	matrix_norm, dic = DecomposeMatrix(matrix)
	fft = np.fft.fftshift(np.fft.fft2(data))
	rextent = GetReciExtent(size=(Lx, Ly), pixels=(N, M))

	# new dimensions: pixels

	C1 = matrix_norm@np.array([0,0])
	C2 = matrix_norm@np.array([0,M-1])
	C3 = matrix_norm@np.array([N-1,M-1])
	C4 = matrix_norm@np.array([N-1,0])

	Nwarp = int(max(C1[0], C2[0], C3[0], C4[0]) - min(C1[0], C2[0], C3[0], C4[0]))
	Mwarp = int(max(C1[1], C2[1], C3[1], C4[1]) - min(C1[1], C2[1], C3[1], C4[1]))

	# new dimensions: size

	X1 = matrix@np.array([0, 0])
	X2 = matrix@np.array([0, Ly])
	X3 = matrix@np.array([Lx,Ly])
	X4 = matrix@np.array([Lx,0])

	LxWarp = max(X1[0], X2[0], X3[0], X4[0]) - min(X1[0], X2[0], X3[0], X4[0])
	LyWarp = max(X1[1], X2[1], X3[1], X4[1]) - min(X1[1], X2[1], X3[1], X4[1])
	extentWarp = 0, LxWarp, 0, LyWarp

	# pixel translation vector

	tx = -int(min(C1[0], C2[0], C3[0], C4[0]))
	ty = -int(min(C1[1], C2[1], C3[1], C4[1]))

	# make the special 2x3 matrix (matrix_norm "+" the translation vector)

	matrixWarp = np.zeros(shape=(2,3))
	matrixWarp[0:2,0:2] = matrix_norm
	matrixWarp[:,2][0] = tx
	matrixWarp[:,2][1] = ty

	# Warp real space image itself

	dataWarp = cv2.warpAffine(data, matrixWarp, (Nwarp, Mwarp), borderMode=cv2.BORDER_CONSTANT)

	# FFT warp of raw data (sometimes better - less sharp but without edge-related artifacts)
	
	matrix_reci = np.linalg.inv(matrix.T)
	matrix_reci_norm, dic = DecomposeMatrix(matrix_reci)
	
	# Finding the correct translation vector for FFTWarp

	# R1 = matrix_reci_norm@np.array([0,0])
	# R2 = matrix_reci_norm@np.array([0,M])
	# R3 = matrix_reci_norm@np.array([N,M])
	# R4 = matrix_reci_norm@np.array([N,0])

	R1 = matrix_reci_norm@np.array([0,0])
	R2 = matrix_reci_norm@np.array([0,M-1])
	R3 = matrix_reci_norm@np.array([N-1,M-1])
	R4 = matrix_reci_norm@np.array([N-1,0])

	txr = -int(min(R1[0], R2[0], R3[0], R4[0]))
	tyr = -int(min(R1[1], R2[1], R3[1], R4[1]))

	# Pixels of FFTWarp

	nr = int(max(R1[0], R2[0], R3[0], R4[0]) - min(R1[0], R2[0], R3[0], R4[0]))
	mr = int(max(R1[1], R2[1], R3[1], R4[1]) - min(R1[1], R2[1], R3[1], R4[1]))

	# Size of FFTWarp

	kxmin, kxmax, kymin, kymax = rextent

	K1 = matrix_reci@np.array([kxmin, kymin])
	K2 = matrix_reci@np.array([kxmax, kymin])
	K3 = matrix_reci@np.array([kxmax, kymax])
	K4 = matrix_reci@np.array([kxmin, kymax])

	kxminWarp = min(K1[0], K2[0], K3[0], K4[0])
	kxmaxWarp = max(K1[0], K2[0], K3[0], K4[0])
	kyminWarp = min(K1[1], K2[1], K3[1], K4[1])
	kymaxWarp = max(K1[1], K2[1], K3[1], K4[1])

	rextentWarp = kxminWarp, kxmaxWarp, kyminWarp, kymaxWarp

	# make the special 2x3 matrix (matrix_reci_norm "+" the translation vector)

	matrix_reciWarp = np.zeros(shape=(2,3))
	matrix_reciWarp[0:2,0:2] = matrix_reci_norm
	matrix_reciWarp[:,2][0] = txr
	matrix_reciWarp[:,2][1] = tyr

	fft_real = cv2.warpAffine(fft.real, matrix_reciWarp, (nr, mr), borderMode=cv2.BORDER_CONSTANT)
	fft_imag = cv2.warpAffine(fft.imag, matrix_reciWarp, (nr, mr), borderMode=cv2.BORDER_CONSTANT)

	fft_real[fft_real==0] = np.nan
	fft_imag[fft_imag==0] = np.nan

	fftWarp = fft_real + 1j*fft_imag

	# return

	return dataWarp, extentWarp, fftWarp, rextentWarp












