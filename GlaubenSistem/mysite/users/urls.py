from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("Glauben/", views.Glauben),
    path("login/", views.login_view),
    path("GlaubenLogin/", views.GlaubenLogin_view),
    path("sidebarGlauben/", views.login_Sidebar)
]
