from setuptools import setup

setup(
    name='papercrop',
    version='0.1',
    description='Combine images of documents in a pdf.',
    
    author='Christian Brugger',
    author_email='brugger.chr@gmail.com',
    url='https://github.com/christianbrugger/pagecrop',
    
    packages=['papercrop'],
    entry_points={"console_scripts": ["crop = papercrop.crop:main"]},
    install_requires=["numpy", "opencv-python", "img2pdf"],
)
