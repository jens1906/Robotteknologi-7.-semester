from glob import glob

from setuptools import find_packages, setup

package_name = 'go2_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jens',
    maintainer_email='jens1906@gmail.com',
    description='TF for the Go2 (real robot and sim)',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'odom_tf = go2_bringup.odom_tf_node:main',
        ],
    },
)
