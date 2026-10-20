from setuptools import find_packages, setup

package_name = 'kpr'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'LICENSE', 'NOTICE.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jens',
    maintainer_email='jens1906@gmail.com',
    description='KPR person re-identification, inference only',
    license='Hippocratic License 3.0',
)
