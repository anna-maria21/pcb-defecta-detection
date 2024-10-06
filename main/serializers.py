from rest_framework import serializers

from main.models import PcbImage


class ProcessedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = PcbImage
        fields = '__all__'
