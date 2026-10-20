from glob import glob

from setuptools import find_packages, setup

package_name = 'go2_scenarios'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/scenes', glob('scenes/*.yaml')),
        ('share/' + package_name + '/models', ['models/walk.dae']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@todo.com',
    description='Worlds with obstacles and a walking leader for the Go2 sim',
    license='BSD',
    entry_points={
        'console_scripts': [
            'scene_runner = go2_scenarios.scene_runner:main',
        ],
    },
)
