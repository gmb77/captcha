<?php
include_once "CaptchaConfig.php";

class CaptchaGenerator
{
    public CaptchaConfig $config;

    public function __construct()
    {
        $this->config = new CaptchaConfig();
    }

    public function generate(): array
    {
        $word = $this->generate_word();
        return $this->generate_image($word);
    }

    private function generate_word(): string
    {
        $word = "";
        $is_vowel = rand(0, 1);
        for ($i = 0; $i < $this->config->char_num; $i++) {
            $pos = $is_vowel ? rand(0, strlen($this->config->chars_vow) - 1) : rand(0, strlen($this->config->chars_cons) - 1);
            $char = $is_vowel ? $this->config->chars_vow[$pos] : $this->config->chars_cons[$pos];
            $word .= $char;
            $is_vowel = !$is_vowel;
        }
        return $word;
    }

    private function generate_image($word): array
    {
        $img = imagecreatetruecolor($this->config->img_width, $this->config->img_height);
        $white = imagecolorallocate($img, 255, 255, 255);
        imagefill($img, 0, 0, $white);

        $chars = $this->initialize_chars($img, $word);
        $this->write_chars($img, $chars, 0);

        $left_border = $this->find_border($img, $white, true);
        $right_border = $this->find_border($img, $white, false);
        $x = ($this->config->img_width - ($right_border - 1 - $left_border)) / 2 - ($left_border + 1);
        imagedestroy($img);

        $img = imagecreatetruecolor($this->config->img_width, $this->config->img_height);
        $bg = $this->config->bg_colour;
        $bg_colour = $this->config->is_colourful ? imagecolorallocate($img, $bg[0], $bg[1], $bg[2]) : $white;
        imagefill($img, 0, 0, $bg_colour);
        strtolower($this->config->img_format) !== "png" or imagecolortransparent($img, $bg_colour);

        $this->config->brush_size < 1 or $this->paint_noise($img);
        $this->write_chars($img, $chars, $x);
        if ($this->config->border_size > 0) {
            $frame_colour = imagecolorallocate($img, $bg[0] * 3 / 4, $bg[1] * 3 / 4, $bg[2] * 3 / 4);
            imagerectangle($img, 0, 0, $this->config->img_width - 1, $this->config->img_height - 1, $frame_colour);
        }

        $captcha["name"] = $word;
        $captcha["extension"] = $this->config->img_format;
        $captcha["picture"] = $img;
        return $captcha;
    }

    private function initialize_chars($img, $word): array
    {
        $black = imagecolorallocate($img, 0, 0, 0);
        $fonts = $this->read_dir(realpath($this->config->font_dir), "ttf");
        if (!count($fonts))
            throw new RuntimeException("Font files are not found.");

        $chars = [];
        for ($i = 0; $i < strlen($word); $i++) {
            $chars[$i]["size"] = rand($this->config->char_size_min, $this->config->char_size_max);
            $chars[$i]["angle"] = rand(0, 1) ? rand(0, $this->config->char_angle_max) : rand(360 - $this->config->char_angle_max, 360);
            $chars[$i]["start_x"] = ($i + 1) * $this->config->char_space;
            $chars[$i]["start_y"] = $this->config->img_height / 2 + rand(0, ($this->config->img_height / 5));
            $chars[$i]["font"] = $this->config->font_dir . DIRECTORY_SEPARATOR . $fonts[array_rand($fonts)];
            $chars[$i]["letter"] = $word[$i];
            if ($this->config->is_colourful) {
                $colour = $this->random_colour();
                if (function_exists("imagecolorallocatealpha"))
                    $chars[$i]["colour"] = imagecolorallocatealpha($img, $colour[0], $colour[1], $colour[2], 10);
                else $chars[$i]["colour"] = imagecolorallocate($img, $colour[0], $colour[1], $colour[2]);
            } else {
                $chars[$i]["colour"] = $black;
            }
        }
        return $chars;
    }

    private function write_chars($img, $chars, $offset)
    {
        foreach ($chars as $char)
            imagettftext($img, $char["size"], $char["angle"], $offset + $char["start_x"], $char["start_y"], $char["colour"], $char["font"], $char["letter"]);
    }

    private function set_brush($img): int
    {
        $brush_colour = $this->config->is_colourful ? imagecolorallocate($img, rand(0, 255), rand(0, 255), rand(0, 255)) :
            imagecolorallocate($img, 0, 0, 0);
        if ($this->config->brush_size > 1 and function_exists("imagesetbrush")) {
            $brush = imagecreatetruecolor($this->config->brush_size, $this->config->brush_size);
            imagefill($brush, 0, 0, $brush_colour);
            imagesetbrush($img, $brush);
            $brush_colour = IMG_COLOR_BRUSHED;
        }
        return $brush_colour;
    }

    private function paint_noise($img)
    {
        for ($i = 0; $i < $this->config->point_num; $i++)
            imagesetpixel($img, rand(0, $this->config->img_width - 1), rand(0, $this->config->img_height - 1), $this->set_brush($img));
        for ($i = 0; $i < $this->config->line_num; $i++)
            imageline($img, rand(0, $this->config->img_width - 1), rand(0, $this->config->img_height - 1),
                rand(0, $this->config->img_width - 1), rand(0, $this->config->img_height - 1), $this->set_brush($img));
        for ($i = 0; $i < $this->config->circle_num; $i++)
            imagearc($img, rand(0, $this->config->img_width - 1), rand(0, $this->config->img_height - 1),
                $radius = rand(5, $this->config->img_width / 3), $radius, 0, 360, $this->set_brush($img));
    }

    private function find_border($img, $bg, $to_right): int
    {
        $sx = imagesx($img);
        $sy = imagesy($img);
        $diff_x = $to_right ? 1 : -1;
        $x = $to_right ? 0 : $sx - 1;
        $y1 = $sy / 3;
        $y2 = $sy * 2 / 3;
        while ($f1 = imagecolorat($img, $x, $y1) === $bg and imagecolorat($img, $x, $y2) === $bg)
            $x += $diff_x;
        $y = $f1 ? $y2 : $y1;
        do {
            $to_up = imagecolorat($img, $x - $diff_x, $y + 1) !== $bg;
            $diff_y = $to_up ? 1 : -1;
            do {
                $x -= $diff_x;
                $y += $diff_y;
            } while ($x >= 0 and $x < $sx and $y >= 0 and $y < $sy and imagecolorat($img, $x, $y) !== $bg);
            for ($y = 0; $y < $sy and imagecolorat($img, $x, $y) === $bg; $y++) null;
        } while ($y < $sy);
        return $x;
    }

    private function random_colour(): array
    {
        do {
            $R = rand(0, 255);
            $G = rand(0, 255);
            $B = rand(0, 255);
        } while ($R + $G + $B >= 400);
        return [$R, $G, $B];
    }

    private function read_dir($dir, $ext): array
    {
        if (!is_dir($dir))
            throw new RuntimeException($dir . " is not found.");
        $files = [];
        $handler = opendir($dir);
        while ($file = readdir($handler))
            !(preg_match("/(?i)\." . $ext . "$/", $file)) or $files[] = $file;
        closedir($handler);
        return $files;
    }
}
