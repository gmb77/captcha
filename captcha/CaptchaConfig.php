<?php


class CaptchaConfig
{
    public string $chars_cons = "BCDFGHKLMNPRTVWXZ";
    public string $chars_vow = "AEIOUY";
    public int $char_num = 4;
    public int $char_space = 20;
    public int $char_size_min = 14;
    public int $char_size_max = 16;
    public int $char_angle_max = 25;

    public string $font_dir = "fonts";
    public string $train_dir = "train_set";
    public string $test_dir = "test_set";
    public string $img_format = "png";
    public int $img_width = 130;
    public int $img_height = 40;

    public array $bg_colour = [255, 255, 255];
    public bool $is_colourful = true;
    public int $border_size = 1;
    public int $brush_size = 1;
    public int $point_num = 10;
    public int $line_num = 1;
    public int $circle_num = 1;
}
