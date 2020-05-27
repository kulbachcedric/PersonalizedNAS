"""personalized_neural_architecture_search URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from django.views.static import serve

from personalization_app import views
from personalized_neural_architecture_search import settings

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'upload', views.upload, name='upload'),
    url(r'list_datasets', views.list_datasets, name='datasets'),
    url(r'^dataset/(.*)/', views.dataset, name='dataset'),
    url(r'^personalize/(.*)/', views.personalize, name='personalize'),
    url(r'^experiment/(.*)/', views.experiment, name='experiment'),
    url(r'^$', views.index, name='index'),
] + staticfiles_urlpatterns()

if settings.DEBUG:
    urlpatterns += [
        url(r'^db_data/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT, }),
    ]