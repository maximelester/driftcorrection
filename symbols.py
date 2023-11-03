#symbols.py


# greek letters (small)

alpha = u'\u03B1'
Alpha = u'\u0391'
beta = u'\u03B2'
Beta = u'\u0392'
gamma = u'\u03B3'
Gamma = u'\u0393'
delta = u'\u03B4'
Delta = u'\u0394'
epsilon = u'\u03B5'
Epsilon = u'\u0395'
zeta = u'\u03B6'
Zeta = u'\u0396'
eta = u'\u03B7'
Eta = u'\u0397'
theta = u'\u03B8'
Theta = u'\u0398'
iota = u'\u03B9'
Iota = u'\u0399'
kappa = u'\u03BA'
Kappa = u'\u039A'
lambdaa = u'\u03BB'
Lambdaa = u'\u039B'
mu = u'\u03BC'
Mu = u'\u039C'
nu = u'\u03BD'
Nu = u'\u039D'
xi = u'\u03BE'
Xi = u'\u039E'
omicron = u'\u03BF'
Omicron = u'\u039F'
pi = u'\u03C0'
Pi = u'\u03A0'
rho = u'\u03C1'
Rho = u'\u03A1'
sigma = u'\u03C3'
Sigma = u'\u03A3'
tau = u'\u03C4'
Tau = u'\u03A4'
upsilon = u'\u03C5'
Upsilon = u'\u03A5'
phi = u'\u03D5' # better version
Phi = u'\u03C6'
chi = u'\u03C7'
Chi = u'\u03A7'
psi = u'\u03C8'
Psi = u'\u03A8'
omega = u'\u03C9'
Omega = u'\u03A9'


e_acute = u'\u00E9'
pm = u'\u00B1'

degree = u'\u00B0'
angstrom = u'\u00C5'
rightarrow = u'\u2192'

superscript_minus = u'\u207B'
superscript_minus1 = superscript_minus+u'\u00B9'
superscript_minus2 = superscript_minus+u'\u00B2'
superscript_2 = u'\u00B2'
superscript_n = u'\u207F'
transpose = u'\u1D40'

subscript_0 = u'\u2080'
subscript_1 = u'\u2081'
subscript_2 = u'\u2082'
subscript_3 = u'\u2083'
subscript_4 = u'\u2084'
subscript_5 = u'\u2085'
subscript_6 = u'\u2086'
subscript_7 = u'\u2087'
subscript_8 = u'\u2088'
subscript_9 = u'\u2089'

subscript_i = u'\u1D62'
subscript_m = u'\u2098'
subscript_n = u'\u2099'
subscript_x = u'\u2093'
subscript_y = u'\u1D67'

subscript_minus = u'\u208B'

times = u'\u2A2F'
bar = u'\u0305' # to type just before the barred character

def replace(s):

	s = s.replace(r'$\alpha$', alpha)
	s = s.replace(r'$\beta$', beta)
	s = s.replace(r'$\gamma$', gamma)
	s = s.replace(r'$\delta$', delta)
	s = s.replace(r'$\epsilon$', epsilon)
	s = s.replace(r'$\zeta$', zeta)
	s = s.replace(r'$\iota$', iota)
	s = s.replace(r'$\kappa$', kappa)
	s = s.replace(r'$\lambda$', lambdaa)
	s = s.replace(r'$\mu$', mu)
	s = s.replace(r'$\nu$', nu)
	s = s.replace(r'$\xi$', xi)
	s = s.replace(r'$\omicron$', omicron)
	s = s.replace(r'$\pi$', pi)
	s = s.replace(r'$\rho$', rho)
	s = s.replace(r'$\sigma$', sigma)
	s = s.replace(r'$\tau$', tau)
	s = s.replace(r'$\upsilon$', upsilon)
	s = s.replace(r'$\phi$', phi)
	s = s.replace(r'$\chi$', chi)
	s = s.replace(r'$\psi$', psi)
	s = s.replace(r'$\omega$', omega)
	s = s.replace('_2', subscript_2)
	s = s.replace('_3', subscript_3)
	s = s.replace('_4', subscript_4)
	s = s.replace('_5', subscript_5)
	s = s.replace('_6', subscript_6)
	s = s.replace('_7', subscript_7)
	s = s.replace('_8', subscript_8)
	s = s.replace('_9', subscript_9)
	s = s.replace('_x', subscript_x)
	#s = s.replace('_y', subscript_y)
	s = s.replace('_{1-x}', subscript_1+subscript_minus+subscript_x)
	s = s.replace('_{x}', subscript_x)

	return s


def subscript(s):

	if s == '0':
		return subscript_0
	elif s == '1':
		return subscript_1
	elif s == '2':
		return subscript_2
	elif s == '3':
		return subscript_3
	elif s == '4':
		return subscript_4
	elif s == '5':
		return subscript_5
	elif s == '6':
		return subscript_6
	elif s == '7':
		return subscript_7
	elif s == '8':
		return subscript_8
	elif s == '9':
		return subscript_9


def replace_inverse(s):

	#print(f'before: {s}')

	s = s.replace(alpha, r'$\alpha$')
	s = s.replace(beta, r'$\beta$')
	s = s.replace(gamma, r'$\gamma$')
	s = s.replace(delta, r'$\delta$')
	s = s.replace(epsilon, r'$\epsilon$')
	s = s.replace(zeta, r'$\zeta$')
	s = s.replace(iota, r'$\iota$')
	s = s.replace(kappa, r'$\kappa$')
	s = s.replace(lambdaa, r'$\lambda$')
	s = s.replace(mu, r'$\mu$')
	s = s.replace(nu, r'$\nu$')
	s = s.replace(xi, r'$\xi$')
	s = s.replace(omicron, r'$\omicron$')
	s = s.replace(pi, r'$\pi$')
	s = s.replace(rho, r'$\rho$')
	s = s.replace(sigma, r'$\sigma$')
	s = s.replace(tau, r'$\tau$')
	s = s.replace(upsilon, r'$\upsilon$')
	s = s.replace(phi, r'$\phi$')
	s = s.replace(chi, r'$\chi$')
	s = s.replace(psi, r'$\psi$')
	s = s.replace(omega, r'$\omega$')
	s = s.replace(subscript_2, '_2')
	s = s.replace(subscript_3, '_3')
	s = s.replace(subscript_4, '_4')
	s = s.replace(subscript_5, '_5')
	s = s.replace(subscript_6, '_6')
	s = s.replace(subscript_7, '_7')
	s = s.replace(subscript_8, '_8')
	s = s.replace(subscript_9, '_9')
	s = s.replace(subscript_x, '_x')
	#s = s.replace(subscript_y, '_y')
	s = s.replace(subscript_1+subscript_minus+subscript_x, '_{1-x}')
	s = s.replace(subscript_x, '_{x}')

	#print(f'before: {s}')


	return s


	
