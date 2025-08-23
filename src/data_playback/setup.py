from setuptools import setup

package_name = 'data_playback'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/multi_agent_playback.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='you@example.com',
    description='Multi-agent dataset playback for collaborative SLAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
           'console_scripts': [
        'imu_listener = data_playback.imu_listener:main',
        'generic_listener = data_playback.generic_listener:main',
    ],
    },
)

