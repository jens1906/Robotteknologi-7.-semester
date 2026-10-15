from setuptools import find_packages, setup

package_name = 'go2_robot_emulator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@todo.com',
    description='Publishes the physical Go2 ROS interface from the Gazebo sim',
    license='BSD',
    entry_points={
        'console_scripts': [
            'sport_api = go2_robot_emulator.sport_api:main',
            'state = go2_robot_emulator.state:main',
            'lidar = go2_robot_emulator.lidar:main',
            'video = go2_robot_emulator.video:main',
            'realsense = go2_robot_emulator.realsense:main',
        ],
    },
)
