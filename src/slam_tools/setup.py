# ~/cslam_ws/src/slam_tools/setup.py
from setuptools import setup, find_packages

package_name = 'slam_tools'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='SLAM helper tools (GT + calibration + alignment)',
    entry_points={
        'console_scripts': [
            'gt_publisher_s3e = slam_tools.gt_publisher_s3e:main',
            'static_tf_from_yaml = slam_tools.static_tf_from_yaml:main',
            'gt_align_once = slam_tools.gt_align_once:main',
        ],
    },
)

