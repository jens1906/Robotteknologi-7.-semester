from glob import glob

from setuptools import find_packages, setup

package_name = 'go2_sim2d'

setup(
    name=package_name,
    version='0.1.0',
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
    description='Top-down 2D Go2 sim with fake perception, for testing the navigation stack',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim2d = go2_sim2d.sim_node:main',
            'fake_perception = go2_sim2d.fake_perception:main',
            'viewer = go2_sim2d.viewer:main',
        ],
    },
)
