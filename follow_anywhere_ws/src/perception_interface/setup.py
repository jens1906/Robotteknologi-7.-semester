from glob import glob

from setuptools import find_packages, setup

package_name = 'perception_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jens',
    maintainer_email='jens1906@gmail.com',
    description='Viewer for perception: compressed camera feed with the tracked people and the leader drawn on it; click a person to calibrate the leader.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # perception_viewer.main does not exist yet: `ros2 run` / the launch file fail until it is written
            'perception_viewer = perception_interface.perception_viewer:main',
        ],
    },
)
