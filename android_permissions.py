from kivy.utils import platform

def check_camera_permission():
    if platform == 'android':
        from android.permissions import request_permissions, Permission
        request_permissions([
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE,
            Permission.INTERNET
        ])
