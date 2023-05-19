from setuptools import setup, find_namespace_packages, find_packages

setup(name='Frodo',
      packages=find_namespace_packages(include=["frodo", "frodo.*"]),
      version='0.0.1',
      description='Frodo. An Easy-to-Use and Extendable Federated Learning Framework.',
      url='https://github.com/CaesarYangs/Frodo',
      author='CaesarYang',
      author_email='caesaryangs@gmail.com',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
          #   "torch>1.10.0",
          #   "tqdm",
          #   "dicom2nifti",
          #   "scikit-image>=0.14",
          #   "medpy",
          #   "scipy",
          #   "batchgenerators>=0.23",
          #   "numpy",
          #   "scikit-learn",
          #   "SimpleITK",
          #   "pandas",
          #   "requests",
          #   "nibabel",
          #   "tifffile",
          #   "matplotlib",
      ],
      entry_points={
          'console_scripts': [
              'frodo_data_construction = frodo.data_construction:main',
              'frodo_preprocess_and_plan = frodo.preprocess_and_plan:main',
              'frodo_train = frodo.train:main'
          ],
      },
      keywords=['deep learning']
      )

