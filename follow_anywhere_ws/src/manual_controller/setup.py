from setuptools import find_packages, setup

package_name = 'manual_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jens',
    maintainer_email='jens1906@gmail.com',
    description='Keyboard (WASD/arrows) manual control of the Unitree Go2',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'keyboard_teleop = manual_controller.keyboard_teleop_node:main',
        ],
    },
)
