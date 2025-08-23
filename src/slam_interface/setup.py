from setuptools import setup

package_name = 'slam_interface'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='SLAM agent interface node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent_node = slam_interface.agent_node:main',
        ],
    },
)

