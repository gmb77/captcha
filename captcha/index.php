<?php
include_once "CaptchaGenerator.php";
$generator = new CaptchaGenerator();
$is_training = $_GET["is_training"] === "1" || strtolower($_GET["is_training"]) === "true";
$dir = realpath(".") . DIRECTORY_SEPARATOR . ($is_training ? $generator->config->train_dir : $generator->config->test_dir);

function save($captcha, $dir)
{
    is_dir($dir) or mkdir($dir);
    $file = $dir . DIRECTORY_SEPARATOR . $captcha["name"] . "." . $captcha["extension"];
    $data = $captcha["picture"];
    switch ($captcha["extension"]) {
        case "jpg":
            imagejpeg($data, $file, 80);
            break;
        case "gif":
            imagegif($data, $file);
            break;
        case "png":
        default:
            imagepng($data, $file);
    }
    imagedestroy($data);
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>CaptchaGen</title>
    <link rel="icon" type="image/png" href="captcha.png">
</head>
<body>
<?php
$num = max($_GET["captcha_num"], 1);
for ($i = 0; $i < $num; $i++)
    save($generator->generate(), $dir);
echo ($is_training ? "Training" : "Test") . " set of " . $num . ($num > 1 ? " images" : " image") . " generated successfully into folder ";
?>
<a href="file://<?php echo $dir; ?>"><?php echo $dir; ?></a><?php echo "."; ?><br/>
</body>
</html>
