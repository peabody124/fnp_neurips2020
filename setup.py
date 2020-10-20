from distutils.core import setup

setup(
    name='tuning_manifold',
    version='0.2',
    packages=['tuning_manifold',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description="Training tuning manifold models",
    install_requires=['tensorflow-datasets', 'tensorflow>=2.2.0', 'tensorflow-probability>=0.10', 'scikit-image', 'absl-py', 'google-cloud-storage', 'datajoint']
)
